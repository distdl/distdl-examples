import time

import torch
from data_loader import get_data_loaders
from network import generate_sequential_network

torch.manual_seed(0)
torch.set_num_threads(32)

# Intel challenge:
num_classes = 6
train_data_dir = 'intel_challenge/seg_train/'
test_data_dir = 'intel_challenge/seg_test/'

resnet_sequential = generate_sequential_network(num_classes)

parameters = [p for p in resnet_sequential.parameters()]
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters, lr=1e-3)

max_batch_size = 64

n_epochs = 10

loud = True

training_loader, test_loader = get_data_loaders(max_batch_size,
                                                train_data_dir,
                                                test_data_dir,
                                                download=True,
                                                dummy=False)

# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/run.py
loss_list, batch_list = [], []

tt = time.time()
for epoch in range(n_epochs):
    tte = time.time()

    for i, (images, labels) in enumerate(training_loader):

        ttp = time.time()

        optimizer.zero_grad()

        output = resnet_sequential(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if loud and i % 1 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {loss_list[-1]}, time: {time.time() - ttp}')

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} time: {time.time() - tte}")

print(f"Total time: {time.time() - tt}")

total = 0
total_correct = 0
avg_loss = 0.0

for i, (images, labels) in enumerate(test_loader):

    total += max_batch_size
    output = resnet_sequential(images)
    avg_loss += criterion(output, labels).sum()
    pred = output.detach().max(1)[1]
    total_correct += pred.eq(labels.view_as(pred)).sum()

perc = float(total_correct) / float(total)

print(f"Total Correct: {total_correct}, Total: {total}, %: {perc}")
