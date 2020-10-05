import sys

import torch
from data_loader import get_data_loaders
from mpi4py import MPI
from network import generate_distributed_network

# Intel challenge:
num_classes = 6
train_data_dir = 'intel_challenge/seg_train/'
test_data_dir = 'intel_challenge/seg_test/'

vgg_distributed = generate_distributed_network(num_classes)

P_base = vgg_distributed.P_base

max_batch_size = 64

n_epochs = 10

loud = True

MPI.COMM_WORLD.Barrier()

parameters = [p for p in vgg_distributed.parameters()]
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters, lr=1e-3)

if P_base.rank == 0:
    training_loader, test_loader = get_data_loaders(max_batch_size,
                                                    train_data_dir,
                                                    test_data_dir,
                                                    download=False,
                                                    dummy=False)
else:
    training_loader, test_loader = get_data_loaders(max_batch_size,
                                                    train_data_dir,
                                                    test_data_dir,
                                                    download=False,
                                                    dummy=True)

# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/run.py
loss_list, batch_list = [], []

tt = MPI.Wtime()
for epoch in range(n_epochs):
    tte = MPI.Wtime()

    for i, (images, labels) in enumerate(training_loader):

        ttp = MPI.Wtime()

        optimizer.zero_grad()

        output = vgg_distributed(images)

        # Compute the loss after forward prop. For now, we do this on rank 0 because
        # CrossEntropy is nonlinear, meaning it cannot be simply sumreduced across
        # all ranks to acheive the correct results.
        if P_base.rank == 0:
            loss = criterion(output, labels)
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i + 1)
            if loud and i % 1 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss_list[-1]}, time: {MPI.Wtime() - ttp}')
                sys.stdout.flush()
        else:
            loss = output.clone()

        loss.backward()
        optimizer.step()

    if P_base.rank == 0:
        print(f"Epoch {epoch} time: {MPI.Wtime() - tte}")
        sys.stdout.flush()

if P_base.rank == 0:
    print(f"Total time: {MPI.Wtime() - tt}")
    sys.stdout.flush()

vgg_distributed.eval()

total = 0
total_correct = 0
avg_loss = 0.0

with torch.no_grad():

    for i, (images, labels) in enumerate(test_loader):

        total += max_batch_size
        output = vgg_distributed(images)
        if P_base.rank == 0:
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

            if loud and i % 10 == 0:
                print(f'Test Batch {i}')
                sys.stdout.flush()

perc = float(total_correct) / float(total)

P_base.print_sequential(f"Rank {P_base.rank}, Total Correct: {total_correct}, Total: {total}, %: {perc}")
sys.stdout.flush()
