# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="686" height="847" alt="Screenshot 2026-05-14 155823" src="https://github.com/user-attachments/assets/30faca9f-4342-4ba7-873c-2ba89cfbcf86" />

## DESIGN STEPS
### STEP 1: 
Load the customer segmentation dataset using Pandas and preprocess the data by removing unnecessary columns, handling missing values, and encoding categorical features.

### STEP 2: 
Split the dataset into input features (X) and target labels (y), then divide the data into training and testing sets using train_test_split().

### STEP 3: 
Create a StandardScaler object, fit it with the training data, and normalize both training and testing feature sets.

### STEP 4: 
Convert the processed data into PyTorch tensors and create TensorDataset and DataLoader objects for batch processing.

### STEP 5: 
Build the Neural Network model using PyTorch by defining input, hidden, and output layers with ReLU activation functions.

### STEP 6: 
Define the loss function (CrossEntropyLoss) and optimizer (Adam), then train the neural network model using the training data for multiple epochs.

### STEP 7:
Evaluate the trained model using the testing data and calculate performance metrics such as accuracy, confusion matrix, and classification report.

### STEP 8:
Plot the confusion matrix using Seaborn and Matplotlib to visualize the classification performance.

### STEP 9:
Use the trained neural network model to predict the customer segmentation class for a sample test input and compare the predicted value with the actual value.

## PROGRAM

### Name: Avinesh B

### Register Number: 212224230029

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)

    def forward(self, x):
        #Include your code here
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here
    model.train()
    for epoch in range(epochs):
      for inputs,labels in train_loader:
        optimizer.zero_grad();
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

   if (epoch + 1) % 10 == 0:
         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model =PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)

```

### Dataset Information
<img width="1520" height="281" alt="Screenshot 2026-05-14 154629" src="https://github.com/user-attachments/assets/442ca30c-58fb-4b90-a11e-24e119bf8bbf" />


### OUTPUT

## Confusion Matrix

<img width="786" height="632" alt="Screenshot 2026-05-14 154803" src="https://github.com/user-attachments/assets/d68fdd58-2bf3-44f7-8b25-c0f66184a6eb" />


## Classification Report
<img width="948" height="538" alt="image" src="https://github.com/user-attachments/assets/3553ff45-c1b7-4a61-8e20-2535aac11682" />


### New Sample Data Prediction
<img width="551" height="70" alt="image" src="https://github.com/user-attachments/assets/18220b31-6c82-4388-a582-185ae505094e" />

## RESULT
Thus, a Neural Network Classification model was successfully developed and trained using PyTorch for customer segmentation prediction.
