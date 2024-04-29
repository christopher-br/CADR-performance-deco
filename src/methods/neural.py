# LOAD MODULES

# Standard library
from typing import Callable
import math

# Proprietary
from src.methods.utils.classes import (
    # Standard imports
    TorchDataset,
    ContinuousCATENN,
    ContinuousCATE,
)

from src.methods.utils.drnet_utils import (
    # DRNet imports
    DRNetHeadLayer,
)

from src.methods.utils.vcnet_utils import (
    # VCNet imports
    VCNet_module,
    TR,
    get_iter,
    criterion,
    criterion_TR,
)

from src.methods.utils.scigan_utils import (
    # SCIGAN imports
    equivariant_layer, 
    invariant_layer, 
    sample_dosages, 
    sample_X, 
    sample_Z,
    get_model_predictions,
)

from src.methods.utils.losses import mmd_rbf_nclass

# Third party
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import shutil
import os

class MLP(ContinuousCATENN):
    """
    Multilayer Perceptron (MLP) class that inherits from the ContinuousCATENN class.

    This class represents a Multilayer Perceptron, which is a type of neural network. It inherits 
    from the ContinuousCATENN class.
    """
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_layers: int = 2,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            input_size (int): The size of the input to the network.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_layers (int, optional): The number of layers in the network. Defaults to 2.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(MLP, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_layers = num_layers

        # Structure
        # Shared layers
        self.layers = nn.Sequential(nn.Linear(self.input_size + 2, self.hidden_size)) # +2 for the dose and treatment
        self.layers.append(self.activation)
        # Add additional layers
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(self.activation)
        # Add output layer
        self.layers.append(nn.Linear(self.hidden_size, 1))
        # Sigmoid activation if binary is True
        if self.binary_outcome == True:
            self.layers.append(nn.Sigmoid())

    def forward(
        self, 
        x: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the network using the input tensor `x`, dose tensor `d` and treatment tensor `t`.
        """
        x = torch.cat((x, d, t), dim=1)

        # Feed through layers
        x = self.layers(x)

        return x

class MultiHeadMLP(ContinuousCATENN):
    """
    The MultiHeadMLP class.
    
    The network has individual heads for different treatments.
    """
    def __init__(
        self,
        input_size: int,
        num_treatments: int,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_representation_layers: int = 2,
        num_inference_layers: int = 2,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.
        
        Parameters:
            input_size (int): The size of the input to the network.
            num_treatments (int): The number of treatments.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_representation_layers (int, optional): The number of representation layers in the network. Defaults to 2.
            num_inference_layers (int, optional): The number of inference layers in the network. Defaults to 2.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(MultiHeadMLP, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_representation_layers = num_representation_layers
        self.num_inference_layers = num_inference_layers
        self.num_treatments = num_treatments

        # Structure
        # Shared layers
        self.shared_layers = nn.Sequential(nn.Linear(self.input_size, self.hidden_size))
        self.shared_layers.append(self.activation)
        # Add additional layers
        for i in range(self.num_representation_layers - 1):
            self.shared_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.shared_layers.append(self.activation)

        # Head networks
        self.head_networks = nn.ModuleList()
        for i in range(self.num_treatments):
            # Build network per head
            help_head = nn.Sequential()
            help_head.append(nn.Linear(self.hidden_size + 1, self.hidden_size))
            help_head.append(self.activation)
            for j in range(num_inference_layers - 1):
                help_head.append(nn.Linear(self.hidden_size, self.hidden_size))
                help_head.append(self.activation)
            # Append last layer
            help_head.append(
                nn.Linear(self.hidden_size, 1)
            )
            if self.binary_outcome == True:
                help_head.append(nn.Sigmoid())
            # Append to module list
            self.head_networks.append(help_head)
            
    def forward(self, x, d, t):
        """
        Defines the forward pass through the network.
        
        Passes data through the shared layers and then through the head layers.
        Saves the result according to the correct bin.
        """
        x = self.shared_layers(x)

        # Add d
        hidden = torch.cat((x, d), dim=1)

        # Dump x
        x = torch.zeros((d.shape))

        # Feed through head layers
        for i in range(self.num_treatments):
            head_out = self.head_networks[i](hidden)
            # Set 0, if in wrong head
            x = x + head_out * (t == i)

        return x

class CBRNet(ContinuousCATENN):
    """
    The CBRNet class.
    
    The network clusters observations based on their location in input space and regularizes for distances between clusters in latent space.
    """
    def __init__(
        self,
        input_size: int,
        mmd: Callable = mmd_rbf_nclass,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_representation_layers: int = 2,
        num_inference_layers: int = 2,
        num_cluster: int = 5,
        regularization_ipm: float = 0.5,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.
        
        Parameters:
            input_size (int): The size of the input to the network.
            mmd (Callable): A callable that computes the Maximum Mean Discrepancy.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_representation_layers (int, optional): The number of representation layers in the network. Defaults to 2.
            num_inference_layers (int, optional): The number of inference layers in the network. Defaults to 2.
            num_cluster (int, optional): The number of clusters. Defaults to 5.
            regularization_ipm (float, optional): The regularization strength for the Integral Probability Metric. Defaults to 0.5.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(CBRNet, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_representation_layers = num_representation_layers
        self.num_inference_layers = num_inference_layers
        self.num_cluster = num_cluster
        self.regularization_ipm = regularization_ipm
        self.mmd = mmd

        # Structure
        # Representation learning layers
        self.representation_layers = nn.Sequential()
        self.representation_layers.append(nn.Linear(self.input_size + 1, self.hidden_size)) # +1 for the treatment
        self.representation_layers.append(self.activation)
        # Add layers
        for i in range(self.num_representation_layers - 1):
            self.representation_layers.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.representation_layers.append(self.activation)

        # Head layers
        self.head_layers = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size) # +1 for the dose
        )
        self.head_layers.append(self.activation)
        # Add additional hidden layers
        for i in range(self.num_inference_layers - 1):
            self.head_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.head_layers.append(self.activation)
        # Add output layer
        self.head_layers.append(nn.Linear(self.hidden_size, 1))
        # Sigmoid activation if binary is True
        if self.binary_outcome == True:
            self.head_layers.append(nn.Sigmoid())

    def fit(self, x, y, d, t):
        """
        Fits the network to the given data.
        
        First, a kmeans clustering is performed on the data. Then, the network is trained using the given data.
        """
        # Build clustering fct
        t_data = TorchDataset(x, y, d, t)
        # Concat x and t
        xdt = torch.cat((t_data.x, t_data.d, t_data.t), dim=1)
        # Train kmeans
        kmeans = KMeans(
            n_clusters=self.num_cluster,
            init="k-means++",
            random_state=42,
            n_init="auto",
        ).fit(xdt)

        # Build fct
        def cluster_fct(
            x: torch.FloatTensor, 
            d: torch.FloatTensor, 
            t:torch.FloatTensor,
        ) -> torch.FloatTensor:
            """
            Function that clusters observations based on their location in input space.
            """
            xdt = torch.cat((x, d, t), dim=1)
            cluster = kmeans.predict(xdt)

            return cluster

        self.cluster_fct = cluster_fct

        super().fit(x, y, d, t)

    def forward(self, x, d, t):
        """
        Performs a forward pass through the network using the input tensor `x`, dose tensor `d`, and the treatment tensor `t`.
        """
        x = torch.cat((x, t), dim=1)
        hidden = self.representation_layers(x)

        # Add d to x
        x = torch.cat((hidden, d), dim=1)

        # Feed through layers
        x = self.head_layers(x)

        return x, hidden

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step using the given batch of data.
        """
        x, y, d, t = batch

        cluster = self.cluster_fct(x, d, t)

        y_hat, hidden = self(x, d, t)

        loss_mse = F.mse_loss(y, y_hat)

        loss_ipm = self.mmd(cluster, self.num_cluster, hidden)

        loss = loss_mse + self.regularization_ipm * loss_ipm

        return loss

    def predict(self, x, d, t):
        """
        Predicts the outcome for the given data.
        """
        x = torch.tensor(x, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).reshape(-1, 1)
        t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

        y_hat, _ = self.forward(x, d, t)

        y_hat = y_hat.reshape(-1).detach().numpy()

        return y_hat

class DRNet(ContinuousCATENN):
    """
    The DRNet class.
    
    The network uses a binning approach to estimate the outcome.
    """
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_representation_layers: int = 2,
        num_inference_layers: int = 2,
        num_bins: int = 10,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            input_size (int): The size of the input to the network.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_representation_layers (int, optional): The number of representation layers in the network. Defaults to 2.
            num_inference_layers (int, optional): The number of inference layers in the network. Defaults to 2.
            num_bins (int, optional): The number of bins for the histogram. Defaults to 10.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(DRNet, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_representation_layers = num_representation_layers
        self.num_inference_layers = num_inference_layers
        self.num_bins = num_bins

        # Define binning fct
        bounds = torch.linspace(
            0 - torch.finfo().eps, 1 + torch.finfo().eps, (self.num_bins + 1)
        )

        def binning_fct(d: torch.FloatTensor) -> torch.FloatTensor:
            """
            Function that bins observations based on their factual dose.
            """
            # Define bounds
            bins = torch.bucketize(d, bounds) - 1
            return bins

        self.binning_fct = binning_fct

        # Structure
        # Shared layers
        self.shared_layers = nn.Sequential(nn.Linear(self.input_size + 1, self.hidden_size)) # +1 for the treatment
        self.shared_layers.append(self.activation)
        # Add additional layers
        for i in range(self.num_representation_layers - 1):
            self.shared_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.shared_layers.append(self.activation)

        # Head networks
        self.head_networks = nn.ModuleList()
        for i in range(self.num_bins):
            # Build network per head
            help_head = nn.Sequential()
            for j in range(num_inference_layers):
                help_head.append(
                    DRNetHeadLayer(
                        self.hidden_size, self.hidden_size, activation=self.activation
                    )
                )
            # Append last layer
            help_head.append(
                DRNetHeadLayer(
                    self.hidden_size, 1, activation=self.activation, last_layer=True
                )
            )
            if self.binary_outcome == True:
                help_head.append(nn.Sigmoid())
            # Append to module list
            self.head_networks.append(help_head)

    def forward(self, x, d, t):
        """
        Defines the forward pass through the network.
        
        Passes data through the shared layers and then through the head layers.
        Saves the result according to the correct bin.
        """
        x = torch.cat((x, t), dim=1)
        x = self.shared_layers(x)

        # Add d
        hidden = torch.cat((x, d), dim=1)

        # Dump x
        x = torch.zeros((d.shape))

        # Get bins
        bins = self.binning_fct(d)

        # Feed through head layers
        for i in range(self.num_bins):
            head_out = self.head_networks[i](hidden)
            # Set 0, if in wrong head
            x = x + head_out * (bins == i)

        return x
    
class MultiHeadDRNet(ContinuousCATENN):
    """
    The MultiheadDRNet class.
    
    This class trains individual DRNet models for each of 'num_treatments' different treatments.
    """
    def __init__(
        self,
        num_treatments: int,
        input_size: int,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_treat_representation_layers: int=2,
        num_representation_layers: int = 2,
        num_inference_layers: int = 2,
        num_bins: int = 10,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.
        """
        # Ini the super module
        super(MultiHeadDRNet, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_treat_representation_layers = num_treat_representation_layers
        self.num_representation_layers = num_representation_layers
        self.num_inference_layers = num_inference_layers
        self.num_bins = num_bins
        self.num_treatments = num_treatments
        
        # Define binning fct
        bounds = torch.linspace(
            0 - torch.finfo().eps, 1 + torch.finfo().eps, (self.num_bins + 1)
        )

        def binning_fct(d: torch.FloatTensor) -> torch.FloatTensor:
            """
            Function that bins observations based on their factual dose.
            """
            # Define bounds
            bins = torch.bucketize(d, bounds) - 1
            return bins

        self.binning_fct = binning_fct
        
        # Structure
        # Shared layers
        self.shared_layers = nn.Sequential(nn.Linear(self.input_size, self.hidden_size))
        self.shared_layers.append(self.activation)
        # Add additional layers
        for i in range(self.num_treat_representation_layers - 1):
            self.shared_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.shared_layers.append(self.activation)
        
        # Treatment networks
        self.treatment_networks = nn.ModuleList()
        for i in range(self.num_treatments):
            # Build individual DRNet per treatment
            help_drnet = nn.ModuleList()
            for j in range(self.num_bins):
                # Build network per head
                help_head = nn.Sequential()
                for j in range(num_inference_layers):
                    help_head.append(
                        DRNetHeadLayer(
                            self.hidden_size, self.hidden_size, activation=self.activation
                        )
                    )
                # Append last layer
                help_head.append(
                    DRNetHeadLayer(
                        self.hidden_size, 1, activation=self.activation, last_layer=True
                    )
                )
                if self.binary_outcome == True:
                    help_head.append(nn.Sigmoid())
                # Append to module list
                help_drnet.append(help_head)
            
            self.treatment_networks.append(help_drnet)
        
    def forward(self, x, d, t):
        """
        Defines the forward pass through the network.
        
        Passes data through the shared layers and then through the head layers.
        Saves the result according to the correct bin.
        """
        hidden = self.shared_layers(x)
        # Add d
        hidden = torch.cat((hidden, d), dim=1)
        
        x = torch.zeros((d.shape))
        
        # Get bins
        bins = self.binning_fct(d)
        
        # Feed through treatment networks
        for i in range(self.num_treatments):
            for j in range(self.num_bins):
                head_out = self.treatment_networks[i][j](hidden)
                x = x + head_out * (t == i) * (bins == j)
        
        return x

class VCNet:
    """
    The VCNet class.
    
    Taken from oringinal implementation. with minor modifications to match style of the rest of the repo.
    """
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        batch_size: int = 500,
        num_steps: int = 1000,
        num_grid: int = 10,
        knots: list = [0.33, 0.66],
        degree: int = 2,
        targeted_regularization: bool = True,
        hidden_size: int = 50,
        wd: float = 5e-3,
        tr_wd: float = 5e-3,
        tr_knots: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        tr_degree: int = 2,
        momentum: float = 0.9,
        init_learning_rate: float = 0.0001,
        alpha: float = 0.5,
        tr_init_learning_rate: float = 0.001,
        beta: float = 1.0,
        binary_outcome: bool = False,
        verbose: bool = False,
    ) -> None:
        # Set seed
        torch.manual_seed(42)

        # Save settings
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.input_size = input_size + 1 # +1 for treatment
        self.learning_rate = learning_rate
        # Save hidden_size or calc if float is passed
        if type(hidden_size) == int:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = int(hidden_size * input_size)
        self.num_grid = num_grid
        self.knots = knots
        self.degree = degree
        self.targeted_regularization = targeted_regularization
        self.wd = wd
        self.tr_wd = tr_wd
        self.tr_knots = tr_knots
        self.tr_degree = tr_degree
        self.momentum = momentum
        self.init_learning_rate = init_learning_rate
        self.alpha = alpha
        self.tr_init_learning_rate = tr_init_learning_rate
        self.beta = beta
        self.binary_outcome = binary_outcome
        self.verbose = verbose

    def fit(self, x, y, d, t):
        # Get num epochs
        num_epochs = math.ceil((self.batch_size * self.num_steps)/x.shape[0])

        train_matrix = torch.from_numpy(np.column_stack((d, x, t, y))).float() # Added treatment

        # Define loader
        train_loader = get_iter(train_matrix, self.batch_size, shuffle=True)

        # Define settings
        cfg_density = [
            (self.input_size, self.hidden_size, 1, "relu"),
            (self.hidden_size, self.hidden_size, 1, "relu"),
        ]
        cfg = [
            (self.hidden_size, self.hidden_size, 1, "relu"),
            (self.hidden_size, 1, 1, "id"),
        ]

        # Load model
        self.model = VCNet_module(
            cfg_density, self.num_grid, cfg, self.degree, self.knots, self.binary_outcome
        )
        self.model._initialize_weights()

        if self.targeted_regularization:
            self.TargetReg = TR(self.tr_degree, self.tr_knots)
            self.TargetReg._initialize_weights()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.init_learning_rate,
            momentum=self.momentum,
            weight_decay=self.wd,
            nesterov=True,
        )

        if self.targeted_regularization:
            tr_optimizer = torch.optim.SGD(
                self.TargetReg.parameters(),
                lr=self.tr_init_learning_rate,
                weight_decay=self.tr_wd,
            )

        # Train
        for epoch in tqdm(
            range(num_epochs),
            leave=False,
            desc="Train VCNet",
            disable=not (self.verbose),
        ):
            for idx, (inputs, y) in enumerate(train_loader):
                idx = idx
                d = inputs[:, 0]
                x = inputs[:, 1:]

                # Train with target reg
                if self.targeted_regularization:
                    optimizer.zero_grad()
                    out = self.model.forward(x, d)
                    trg = self.TargetReg(d)
                    loss = criterion(out, y, alpha=self.alpha) + criterion_TR(
                        out, trg, y, beta=self.beta
                    )
                    loss.backward()
                    optimizer.step()

                    tr_optimizer.zero_grad()
                    out = self.model.forward(x, d)
                    trg = self.TargetReg(d)
                    tr_loss = criterion_TR(out, trg, y, beta=self.beta)
                    tr_loss.backward()
                    tr_optimizer.step()
                # Train withouth target reg
                else:
                    optimizer.zero_grad()
                    out = self.model.forward(x, d)
                    loss = criterion(out, y, alpha=self.alpha)
                    loss.backward()
                    optimizer.step()

    def score(self, x, y, d, t):
        y_hat = self.predict(x, d, t)

        mse = ((y - y_hat) ** 2).mean()

        return mse

    def predict(self, x, d, t):
        pred_matrix = torch.from_numpy(np.column_stack((d, x, t, d))).float()
        # Define pred loader
        pred_loader = get_iter(pred_matrix, pred_matrix.shape[0], shuffle=False)

        for idx, (inputs, y) in enumerate(pred_loader):
            # Get inputs
            d = inputs[:, 0]
            x = inputs[:, 1:]

            # Get estimates
            y_hat = self.model.forward(x, d)[1].data.squeeze().numpy()

        return y_hat

class SCIGAN(ContinuousCATE):
    def __init__(
        self,
        input_size: int,
        num_treatments: int,
        export_dir: str="models/scigan",
        hidden_size: int=64,
        hidden_size_inv_eqv: int=None,
        batch_size: int=128,
        num_steps_gan: int=5000,
        num_steps_inf: int=10000,
        alpha: float=1.0,
        num_dosage_samples: int=5,
    ):
        self.num_features = input_size
        self.num_treatments = num_treatments
        self.export_dir = export_dir
        self.h_dim = hidden_size
        self.h_inv_eqv_dim = (hidden_size if hidden_size_inv_eqv is None else hidden_size_inv_eqv) # Take hidden_size if not specified
        self.batch_size = batch_size
        self.num_steps_gan = num_steps_gan
        self.num_steps_inf = num_steps_inf
        self.alpha = alpha
        self.num_dosage_samples = num_dosage_samples

        self.size_z = self.num_treatments * self.num_dosage_samples
        self.num_outcomes = self.num_treatments * self.num_dosage_samples

        tf.reset_default_graph()
        tf.random.set_random_seed(10)

        # Feature (X)
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_features], name='input_features')
        # Treatment (T) - one-hot encoding for the treatment
        self.T = tf.placeholder(tf.float32, shape=[None, self.num_treatments], name='input_treatment')
        # Dosage (D)
        self.D = tf.placeholder(tf.float32, shape=[None, 1], name='input_dosage')
        # Dosage samples (D)
        self.Treatment_Dosage_Samples = tf.placeholder(tf.float32,
                                                       shape=[None, self.num_treatments, self.num_dosage_samples],
                                                       name='input_treatment_dosage_samples')
        # Treatment dosage mask to indicate the factual outcome
        self.Treatment_Dosage_Mask = tf.placeholder(tf.float32,
                                                    shape=[None, self.num_treatments, self.num_dosage_samples],
                                                    name='input_treatment_dosage_mask')
        # Outcome (Y)
        self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='input_y')
        # Random Noise (G)
        self.Z_G = tf.placeholder(tf.float32, shape=[None, self.size_z], name='input_noise')

    def generator(self, x, y, t, d, z, treatment_dosage_samples):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            inputs = tf.concat(axis=1, values=[x, y, t, d, z])
            G_shared = tf.layers.dense(inputs, self.h_dim, activation=tf.nn.elu, name='shared')
            G_treatment_dosage_outcomes = dict()

            for treatment in range(self.num_treatments):
                treatment_dosages = treatment_dosage_samples[:, treatment]
                treatment_dosages = tf.reshape(treatment_dosages, shape=(-1, 1))
                G_shared_expand = tf.reshape(tf.tile(G_shared, multiples=[1, self.num_dosage_samples]),
                                             shape=(-1, self.h_dim))
                input_counterfactual_dosage = tf.concat(axis=1, values=[G_shared_expand, treatment_dosages])

                treatment_layer_1 = tf.layers.dense(input_counterfactual_dosage, self.h_dim, activation=tf.nn.elu,
                                                    name='treatment_layer_1_%s' % str(treatment), reuse=tf.AUTO_REUSE)

                treatment_layer_2 = tf.layers.dense(treatment_layer_1, self.h_dim, activation=tf.nn.elu,
                                                    name='treatment_layer_2_%s' % str(treatment), reuse=tf.AUTO_REUSE)

                treatment_dosage_output = tf.layers.dense(treatment_layer_2, 1, activation=None,
                                                          name='treatment_output_%s' % str(treatment),
                                                          reuse=tf.AUTO_REUSE)

                dosage_counterfactuals = tf.reshape(treatment_dosage_output, shape=(-1, self.num_dosage_samples))

                G_treatment_dosage_outcomes[treatment] = dosage_counterfactuals

            G_logits = tf.concat(list(G_treatment_dosage_outcomes.values()), axis=1)
            G_logits = tf.reshape(G_logits, shape=(-1, self.num_treatments, self.num_dosage_samples))

        return G_logits, G_treatment_dosage_outcomes

    def dosage_discriminator(self, x, y, treatment_dosage_samples, treatment_dosage_mask,
                             G_treatment_dosage_outcomes):

        with tf.variable_scope('dosage_discriminator', reuse=tf.AUTO_REUSE):
            patient_features_representation = tf.expand_dims(tf.layers.dense(x, self.h_dim, activation=tf.nn.elu),
                                                             axis=1)
            D_dosage_outcomes = dict()
            for treatment in range(self.num_treatments):
                treatment_mask = treatment_dosage_mask[:, treatment]
                treatment_dosages = treatment_dosage_samples[:, treatment]
                G_treatment_dosage_outcomes[treatment] = treatment_mask * y + (1 - treatment_mask) * \
                                                         G_treatment_dosage_outcomes[treatment]

                dosage_samples = tf.expand_dims(treatment_dosages, axis=-1)
                dosage_potential_outcomes = tf.expand_dims(G_treatment_dosage_outcomes[treatment], axis=-1)

                inputs = tf.concat(axis=-1, values=[dosage_samples, dosage_potential_outcomes])
                D_h1 = tf.nn.elu(equivariant_layer(inputs, self.h_inv_eqv_dim, layer_id=1,
                                                   treatment_id=treatment) + patient_features_representation)
                D_h2 = tf.nn.elu(equivariant_layer(D_h1, self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment))
                D_logits_treatment = tf.layers.dense(D_h2, 1, activation=None,
                                                     name='treatment_output_%s' % str(treatment))

                D_dosage_outcomes[treatment] = tf.squeeze(D_logits_treatment, axis=-1)

            D_dosage_logits = tf.concat(list(D_dosage_outcomes.values()), axis=-1)
            D_dosage_logits = tf.reshape(D_dosage_logits, shape=(-1, self.num_treatments, self.num_dosage_samples))

        return D_dosage_logits, D_dosage_outcomes

    def treatment_discriminator(self, x, y, treatment_dosage_samples, treatment_dosage_mask,
                                G_treatment_dosage_outcomes):
        with tf.variable_scope('treatment_discriminator', reuse=tf.AUTO_REUSE):
            patient_features_representation = tf.layers.dense(x, self.h_dim, activation=tf.nn.elu)

            D_treatment_outcomes = dict()
            for treatment in range(self.num_treatments):
                treatment_mask = treatment_dosage_mask[:, treatment]
                treatment_dosages = treatment_dosage_samples[:, treatment]
                G_treatment_dosage_outcomes[treatment] = treatment_mask * y + (1 - treatment_mask) * \
                                                         G_treatment_dosage_outcomes[treatment]

                dosage_samples = tf.expand_dims(treatment_dosages, axis=-1)
                dosage_potential_outcomes = tf.expand_dims(G_treatment_dosage_outcomes[treatment], axis=-1)

                inputs = tf.concat(axis=-1, values=[dosage_samples, dosage_potential_outcomes])
                D_treatment_rep = invariant_layer(x=inputs, h_dim=self.h_inv_eqv_dim, treatment_id=treatment)

                D_treatment_outcomes[treatment] = D_treatment_rep

            D_treatment_representations = tf.concat(list(D_treatment_outcomes.values()), axis=-1)
            D_shared_representation = tf.concat([D_treatment_representations, patient_features_representation], axis=-1)

            D_treatment_rep_hidden = tf.layers.dense(D_shared_representation, self.h_dim, activation=tf.nn.elu,
                                                     name='rep_all',
                                                     reuse=tf.AUTO_REUSE)

            D_treatment_logits = tf.layers.dense(D_treatment_rep_hidden, self.num_treatments, activation=None,
                                                 name='output_all',
                                                 reuse=tf.AUTO_REUSE)

        return D_treatment_logits

    def inference(self, x, treatment_dosage_samples):
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            inputs = x
            I_shared = tf.layers.dense(inputs, self.h_dim, activation=tf.nn.elu, name='shared')

            I_treatment_dosage_outcomes = dict()

            for treatment in range(self.num_treatments):
                dosage_counterfactuals = dict()
                treatment_dosages = treatment_dosage_samples[:, treatment]

                for index in range(self.num_dosage_samples):
                    dosage_sample = tf.expand_dims(treatment_dosages[:, index], axis=-1)
                    input_counterfactual_dosage = tf.concat(axis=1, values=[I_shared, dosage_sample])

                    treatment_layer_1 = tf.layers.dense(input_counterfactual_dosage, self.h_dim, activation=tf.nn.elu,
                                                        name='treatment_layer_1_%s' % str(treatment),
                                                        reuse=tf.AUTO_REUSE)

                    treatment_layer_2 = tf.layers.dense(treatment_layer_1, self.h_dim, activation=tf.nn.elu,
                                                        name='treatment_layer_2_%s' % str(treatment),
                                                        reuse=tf.AUTO_REUSE)

                    treatment_dosage_output = tf.layers.dense(treatment_layer_2, 1, activation=None,
                                                              name='treatment_output_%s' % str(treatment),
                                                              reuse=tf.AUTO_REUSE)

                    dosage_counterfactuals[index] = treatment_dosage_output

                I_treatment_dosage_outcomes[treatment] = tf.concat(list(dosage_counterfactuals.values()), axis=-1)

            I_logits = tf.concat(list(I_treatment_dosage_outcomes.values()), axis=1)
            I_logits = tf.reshape(I_logits, shape=(-1, self.num_treatments, self.num_dosage_samples))

        return I_logits, I_treatment_dosage_outcomes

    def fit(self, Train_X, Train_Y, Train_D, Train_T, verbose=False):
        # Save min and max values for y
        self.min_y = np.min(Train_Y)
        self.max_y = np.max(Train_Y)
        
        # Transform y to be between 0 and 1
        Train_Y = (Train_Y - self.min_y) / (self.max_y - self.min_y)
        
        # Remove existing model
        if os.path.exists(self.export_dir):
            shutil.rmtree(self.export_dir)
        # 1. Counterfactual generator
        G_logits, G_treatment_dosage_outcomes = self.generator(x=self.X, y=self.Y, t=self.T, d=self.D,
                                                               z=self.Z_G,
                                                               treatment_dosage_samples=self.Treatment_Dosage_Samples)

        # 2. Dosage discriminator
        D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(x=self.X, y=self.Y,
                                                                       treatment_dosage_samples=self.Treatment_Dosage_Samples,
                                                                       treatment_dosage_mask=self.Treatment_Dosage_Mask,
                                                                       G_treatment_dosage_outcomes=G_treatment_dosage_outcomes)
        # 3. Treatment discriminator
        D_treatment_logits = self.treatment_discriminator(x=self.X, y=self.Y,
                                                          treatment_dosage_samples=self.Treatment_Dosage_Samples,
                                                          treatment_dosage_mask=self.Treatment_Dosage_Mask,
                                                          G_treatment_dosage_outcomes=G_treatment_dosage_outcomes)

        # 4. Inference network
        I_logits, I_treatment_dosage_outcomes = self.inference(self.X, self.Treatment_Dosage_Samples)

        G_outcomes = tf.identity(G_logits, name='generator_outcomes')
        I_outcomes = tf.identity(I_logits, name="inference_outcomes")

        # 1. Dosage discriminator loss
        num_examples = tf.cast(self.batch_size, dtype=tf.int64)
        factual_treatment_idx = tf.argmax(self.T, axis=1)
        idx = tf.stack([tf.range(num_examples), factual_treatment_idx], axis=-1)

        D_dosage_logits_factual_treatment = tf.gather_nd(D_dosage_logits, idx)
        Dosage_Mask = tf.gather_nd(self.Treatment_Dosage_Mask, idx)

        D_dosage_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=Dosage_Mask, logits=D_dosage_logits_factual_treatment))

        # 2. Treatment discriminator loss
        D_treatment_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reduce_max(self.Treatment_Dosage_Mask, axis=-1),
                                                    logits=D_treatment_logits))

        # 3. Overall discriminator loss
        D_combined_prob = tf.nn.sigmoid(D_dosage_logits) * tf.nn.sigmoid(
            tf.tile(tf.expand_dims(D_treatment_logits, axis=-1), multiples=[1, 1, self.num_dosage_samples]))

        D_combined_loss = tf.reduce_mean(
            self.Treatment_Dosage_Mask * -tf.log(D_combined_prob + 1e-7) + (1.0 - self.Treatment_Dosage_Mask) * -tf.log(
                1.0 - D_combined_prob + 1e-7))

        # 4. Generator loss
        G_loss_GAN = -D_combined_loss
        G_logit_factual = tf.expand_dims(tf.reduce_sum(self.Treatment_Dosage_Mask * G_logits, axis=[1, 2]), axis=-1)
        G_loss_R = tf.reduce_mean((self.Y - G_logit_factual) ** 2)
        G_loss = self.alpha * tf.sqrt(G_loss_R) + G_loss_GAN

        # 4. Inference loss
        I_logit_factual = tf.expand_dims(tf.reduce_sum(self.Treatment_Dosage_Mask * I_logits, axis=[1, 2]), axis=-1)
        I_loss1 = tf.reduce_mean((G_logits - I_logits) ** 2)
        I_loss2 = tf.reduce_mean((self.Y - I_logit_factual) ** 2)
        I_loss = tf.sqrt(I_loss1) + tf.sqrt(I_loss2)

        theta_G = tf.trainable_variables(scope='generator')
        theta_D_dosage = tf.trainable_variables(scope='dosage_discriminator')
        theta_D_treatment = tf.trainable_variables(scope='treatment_discriminator')
        theta_I = tf.trainable_variables(scope='inference')

        # Solver
        G_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=theta_G)
        D_dosage_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_dosage_loss, var_list=theta_D_dosage)
        D_treatment_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_treatment_loss,
                                                                                  var_list=theta_D_treatment)
        I_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(I_loss, var_list=theta_I)

        # Setup tensorflow
        tf_device = 'gpu'
        if tf_device == "cpu":
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Iterations
        for it in tqdm(range(self.num_steps_gan), leave=False, desc="Training GAN"):
            for kk in range(2):
                idx_mb = sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb, :]
                T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
                D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
                Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
                Z_G_mb = sample_Z(self.batch_size, self.size_z)

                treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                          self.num_dosage_samples)
                factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
                treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

                treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                        self.num_dosage_samples])
                treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
                treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

                _, G_loss_curr, G_logits_curr, G_logit_factual_curr = self.sess.run(
                    [G_solver, G_loss, G_logits, G_logit_factual],
                    feed_dict={self.X: X_mb, self.T: treatment_one_hot, self.D: D_mb[:, np.newaxis],
                               self.Treatment_Dosage_Samples: treatment_dosage_samples,
                               self.Treatment_Dosage_Mask: treatment_dosage_mask, self.Y: Y_mb,
                               self.Z_G: Z_G_mb})

            for kk in range(1):
                idx_mb = sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb, :]
                T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
                D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
                Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
                Z_G_mb = sample_Z(self.batch_size, self.size_z)

                treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                          self.num_dosage_samples)
                factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
                treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

                treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                        self.num_dosage_samples])
                treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
                treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

                _, D_dosage_loss_curr = self.sess.run([D_dosage_solver, D_dosage_loss],
                                                      feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                                 self.D: D_mb[:, np.newaxis],
                                                                 self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                                 self.Treatment_Dosage_Mask: treatment_dosage_mask,
                                                                 self.Y: Y_mb, self.Z_G: Z_G_mb})

                idx_mb = sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb, :]
                T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
                D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
                Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
                Z_G_mb = sample_Z(self.batch_size, self.size_z)

                treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                          self.num_dosage_samples)
                factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
                treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

                treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                        self.num_dosage_samples])
                treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
                treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

                _, D_treatment_loss_curr = self.sess.run([D_treatment_solver, D_treatment_loss],
                                                         feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                                    self.D: D_mb[:, np.newaxis],
                                                                    self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                                    self.Treatment_Dosage_Mask: treatment_dosage_mask,
                                                                    self.Y: Y_mb, self.Z_G: Z_G_mb})

            if it % 1000 == 0 and verbose:
                D_treatment_loss_curr, D_dosage_loss_curr, G_loss_curr, = self.sess.run(
                    [D_treatment_loss, D_dosage_loss, G_loss],
                    feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                               self.D: D_mb[:, np.newaxis],
                               self.Treatment_Dosage_Samples: treatment_dosage_samples,
                               self.Treatment_Dosage_Mask: treatment_dosage_mask,
                               self.Y: Y_mb, self.Z_G: Z_G_mb})

                print('Iter: {}'.format(it))
                print('D_loss_treatments: {:.4}'.format((D_treatment_loss_curr)))
                print('D_loss_dosages: {:.4}'.format((D_dosage_loss_curr)))
                print('G_loss: {:.4}'.format((G_loss_curr)))
                print()

        # Train Inference Network
        for it in tqdm(range(self.num_steps_inf), leave=False, desc="Training inference network"):
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X[idx_mb, :]
            T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
            D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
            Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
            Z_G_mb = sample_Z(self.batch_size, self.size_z)

            treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                      self.num_dosage_samples)
            factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
            treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

            treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                    self.num_dosage_samples])
            treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
            treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

            _, I_loss_curr = self.sess.run([I_solver, I_loss],
                                           feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                      self.D: D_mb[:, np.newaxis],
                                                      self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                      self.Treatment_Dosage_Mask: treatment_dosage_mask, self.Y: Y_mb,
                                                      self.Z_G: Z_G_mb})

            if it % 1000 == 0 and verbose:
                print('Iter: {}'.format(it))
                print('I_loss: {:.4}'.format((I_loss_curr)))
                print()

        tf.compat.v1.saved_model.simple_save(self.sess, export_dir=self.export_dir,
                                             inputs={'input_features': self.X,
                                                     'input_treatment_dosage_samples': self.Treatment_Dosage_Samples},
                                             outputs={'inference_outcome': I_logits})

    def predict(self, x, d, t):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], self.export_dir)
            
            preds = get_model_predictions(
                sess=sess,
                num_treatments=self.num_treatments,
                num_dosage_samples=self.num_dosage_samples,
                test_data={"x": x, "d": d, "t": t},
            )
        
        # Transform y back to original scale
        preds = preds * (self.max_y - self.min_y) + self.min_y
            
        return preds