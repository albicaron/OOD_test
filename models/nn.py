import torch
import torch.nn as nn
import torch.optim as optim

# Import auroc
from sklearn.metrics import balanced_accuracy_score


# Define the neural network model
class ProbabilisticNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(ProbabilisticNN, self).__init__()

        self.optimizer = None
        self.criterion = None

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def setup_model(self, learning_rate):
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, x_train, y_train, num_epochs, batch_size):
        for epoch in range(num_epochs):
            for i in range(0, len(x_train), batch_size):
                self.optimizer.zero_grad()
                batch_data = x_train[i:i+batch_size]
                target = y_train[i:i+batch_size].long()

                output = self.forward(batch_data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, x):
        return self(x)


    def evaluate_acc(self, x_test, y_test):
        with torch.no_grad():

            # Compute the AUROC
            y_pred = self.predict_proba(x_test)
            y_pred = y_pred[:, 1].numpy()

        return balanced_accuracy_score(y_test, y_pred > 0.5)
