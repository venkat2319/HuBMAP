import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTModel, ViTConfig

class EnsembleModel(nn.Module):
    def __init__(self, num_classes, num_heads):
        super(EnsembleModel, self).__init__()
        
        # Initialize Vision Transformer and EfficientNet models
        self.vit = ViTModel(ViTConfig(hidden_size=768, num_attention_heads=num_heads))
        self.effnet = models.efficientnet_b3(pretrained=True)
        
        # 1x1 convolution layer to combine the features
        self.conv = nn.Conv2d(1536, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Forward pass through Vision Transformer and EfficientNet models
        vit_output = self.vit(x)
        effnet_output = self.effnet(x)
        
        # Concatenate the features and apply 1x1 convolution
        output = torch.cat([vit_output.last_hidden_state, effnet_output], dim=1)
        output = self.conv(output)
        
        return output

# Create an instance of the ensemble model
model = EnsembleModel(num_classes=2, num_heads=8)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Train the model
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Print the training loss after every epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    total_correct = 0
    total_images = 0
    
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_images += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
    print('Accuracy: {:.2f}%'.format(100 * total_correct / total_images))
