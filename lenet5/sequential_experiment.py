import time

import torch
from data_loader import get_data_loaders
from network import generate_sequential_network

lenet5_sequential = generate_sequential_network()

parameters = [p for p in lenet5_sequential.parameters()]
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters, lr=1e-3)

max_batch_size = 256

n_epochs = 10

loud = True

training_loader, test_loader = get_data_loaders(max_batch_size,
                                                download=True,
                                                dummy=False)

# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/run.py
loss_list, batch_list = [], []

tt = time.time()
for epoch in range(n_epochs):
    tte = time.time()

    for i, (images, labels) in enumerate(training_loader):

        optimizer.zero_grad()

        output = lenet5_sequential(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if loud and i % 10 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {loss_list[-1]}')

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} time: {time.time() - tte}")

print(f"Total time: {time.time() - tt}")

total = 0
total_correct = 0
avg_loss = 0.0

for i, (images, labels) in enumerate(test_loader):

    total += max_batch_size
    output = lenet5_sequential(images)
    avg_loss += criterion(output, labels).sum()
    pred = output.detach().max(1)[1]
    total_correct += pred.eq(labels.view_as(pred)).sum()

perc = float(total_correct) / float(total)

print(f"Total Correct: {total_correct}, Total: {total}, %: {perc}")
