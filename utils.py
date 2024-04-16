from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
class Utils:
    @staticmethod
    def visualise_data(MY_X, MY_y):
        # Apply PCA to reduce data to 2 dimensions
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(MY_X)

        # Get target names
        target_names = MY_y

        # Create hover text for labels
        # hover_text = [f'Target: {target_names[label]}<br>Attributes (only first 5): {", ".join(map(str, attrs[0:5]))}'
        #               for label, attrs in zip(y, X)]

        # Create trace for data points
        trace = go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            # hovertext=hover_text,
            marker=dict(
                size=7,
                color=target_names,
                colorscale='Viridis',
                line=dict(
                    color='rgb(0, 0, 0)',
                    width=0.5
                ),
                opacity=0.8
            )
        )

        # Create layout
        layout = go.Layout(
            title='Data Distribution in 2D (PCA)',
            xaxis=dict(title='Principal Component 1'),
            yaxis=dict(title='Principal Component 2'),
            width=800,  # Set the width of the plot
            height=600,  # Set the height of the plot
        )

        # Create figure
        fig = go.Figure(data=[trace], layout=layout)

        # Show interactive plot
        pio.show(fig)

    @staticmethod
    def visualise_data(MY_X, MY_y):
        # Apply PCA to reduce data to 2 dimensions
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(MY_X)

        # Get target names
        target_names = MY_y

        # Create hover text for labels
        # hover_text = [f'Target: {target_names[label]}<br>Attributes (only first 5): {", ".join(map(str, attrs[0:5]))}'
        #               for label, attrs in zip(y, X)]

        # Create trace for data points
        trace = go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            # hovertext=hover_text,
            marker=dict(
                size=7,
                color=target_names,
                colorscale='Viridis',
                line=dict(
                    color='rgb(0, 0, 0)',
                    width=0.5
                ),
                opacity=0.8
            )
        )

        # Create layout
        layout = go.Layout(
            title='Data Distribution in 2D (PCA)',
            xaxis=dict(title='Principal Component 1'),
            yaxis=dict(title='Principal Component 2'),
            width=800,  # Set the width of the plot
            height=600,  # Set the height of the plot
        )

        # Create figure
        fig = go.Figure(data=[trace], layout=layout)

        # Show interactive plot
        pio.show(fig)

    @staticmethod
    def plot_training_validation_metrics(train_accuracies, val_accuracies, train_losses, val_losses, model, optimizer, learning_rate):
        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f'Training and Validation Accuracy',
            f'Training and Validation Loss'
        ))

        # Plot training and validation accuracy
        fig.add_trace(go.Scatter(x=list(range(len(train_accuracies))), y=train_accuracies, mode='lines', name='Train Acc', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(val_accuracies))), y=val_accuracies, mode='lines', name='Val Acc', line=dict(color='orange')), row=1, col=1)

        # Plot training and validation loss
        fig.add_trace(go.Scatter(x=list(range(len(train_losses))), y=train_losses, mode='lines', name='Train Loss', line=dict(color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(val_losses))), y=val_losses, mode='lines', name='Val Loss', line=dict(color='orange')), row=1, col=2)

        # Update layout
        fig.update_layout(title_text=f"Training and Validation Metrics -> Model: {model.__class__.__name__}, Optimizer: {optimizer.__class__.__name__}, Learning Rate: {learning_rate}",
                          title_font=dict(size=18),
                          showlegend=True,
                          title_x=0.5)  # Center title

        # Update subplot titles
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=2)

        # Show plot
        fig.show()

    @staticmethod
    def calculate_metrics(model, X_test, y_test):
        with torch.no_grad():
            y_predicted = model(X_test)
            y_predicted_cls = (y_predicted > 0.5).float()
            acc = accuracy_score(y_test, y_predicted_cls, normalize=True)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_predicted_cls, average='binary')
            cm = confusion_matrix(y_test, y_predicted_cls)
            print(f'Accuracy: {acc:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F-score: {fscore:.4f}')
            labels = ['Negative', 'Positive']  # Assuming binary classification
            fig = go.Figure(data=go.Heatmap(z=cm[::-1], x=labels, y=labels[::-1], colorscale='Blues', showscale=False))
            for i in range(len(labels)):
                for j in range(len(labels)):
                    fig.add_annotation(x=labels[j], y=labels[::-1][i], text=str(cm[::-1][i][j]), showarrow=False, font=dict(color='black', size=14))
            fig.update_layout(title=f'Confusion Matrix -> Model: {model.__class__.__name__}', xaxis_title='Predicted Labels', yaxis_title='True Labels', width=500, height=500)
            fig.show()
            return acc, precision, recall, fscore, cm