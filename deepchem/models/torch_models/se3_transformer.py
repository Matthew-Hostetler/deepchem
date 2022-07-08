"""
SE(3) Invariant Transformer.
"""
from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.se3_transformer_components.fiber import Fiber
from deepchem.models.torch_models.torch_model import TorchModel
from se3_transformer_components.transformer import SE3TransformerInternal
from typing import Literal, Optional
import torch.nn as nn
import torch.nn.functional as F

class SE3Transformer(nn.Module):
  """SE(3) Invariant Transformer for graph property prediction

  Examples
  --------

  >>> import deepchem as dc
  >>> import dgl
  >>> from deepchem.models.torch_models import SE3Transformer
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
  >>> graphs = featurizer.featurize(smiles)
  >>> print(type(graphs[0]))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
  >>> # Batch two graphs into a graph of two connected components
  >>> batch_dgl_graph = dgl.batch(dgl_graphs)
  >>> model = SE3Transformer(n_tasks=1, mode='regression')
  >>> preds = model(batch_dgl_graph)
  >>> print(type(preds))
  <class 'torch.Tensor'>
  >>> preds.shape == (2, 1)
  True

  References
  ----------
  .. [1] Fabian B. Fuchs and Daniel E. Worrall and Volker Fischer and Max Welling.
         "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks". NeurIPS 2020

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and e3nn (https://e3nn.org/) to be installed.
  """

  def __init__(self,
              num_layers: int,
              fiber_in: Fiber,
              fiber_hidden: Fiber,
              fiber_out: Fiber,
              num_heads: int,
              channels_div: int,
              fiber_edge: Fiber = Fiber({}),
              return_type: Optional[int] = None,
              pooling: Optional[Literal['avg', 'max']] = None,
              norm: bool = True,
              use_layer_norm: bool = True,
              low_memory: bool = False,
              **kwargs):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg' or 'max' graph pooling before MLP layers
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param low_memory:          If True, will use slower ops that use less memory
        """
        
  if mode not in ['classification', 'regression']:
    raise ValueError("mode must be either 'classification' or 'regression'")

  super(SE3Transformer, self).__init__()

  self.model = SE3TransformerInternal(
      num_layers=num_layers,
      fiber_in=fiber_in,
      fiber_hidden=fiber_hidden,
      fiber_out=fiber_out,
      num_heads=num_heads,
      channels_div=channels_div,
      fiber_edge=fiber_edge,
      return_type=return_type,
      pooling=pooling,
      norm=norm,
      use_layer_norm=use_layer_norm,
      low_memory=low_memory)

  def forward(self, g):
    """Predict graph labels

    Parameters
    ----------
    g: DGLGraph
      A DGLGraph for a batch of graphs. It stores the node features in
      ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
      ``dgl_graph.edata[self.efeat_name]``.

    Returns
    -------
    torch.Tensor
      The model output.

      * When self.mode = 'regression',
        its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
      * When self.mode = 'classification', the output consists of probabilities
        for classes. Its shape will be
        ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)`` if self.n_tasks > 1;
        its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if self.n_tasks is 1.
    torch.Tensor, optional
      This is only returned when self.mode = 'classification', the output consists of the
      logits for classes before softmax.
    """
    node_feats = g.ndata[self.nfeat_name]
    edge_feats = g.edata[self.efeat_name]
    out = self.model(g, node_feats, edge_feats)

    if self.mode == 'classification':
      if self.n_tasks == 1:
        logits = out.view(-1, self.n_classes)
        softmax_dim = 1
      else:
        logits = out.view(-1, self.n_tasks, self.n_classes)
        softmax_dim = 2
      proba = F.softmax(logits, dim=softmax_dim)
      return proba, logits
    else:
      return out


class SE3TransformerModel(TorchModel):
  """SE(3) Invariant Transformer for graph property prediction

  Examples
  --------
  >>> import deepchem as dc
  >>> from deepchem.models.torch_models import MPNNModel
  >>> # preparing dataset
  >>> smiles = ["C1CCC1", "CCC"]
  >>> labels = [0., 1.]
  >>> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
  >>> X = featurizer.featurize(smiles)
  >>> dataset = dc.data.NumpyDataset(X=X, y=labels)
  >>> # training model
  >>> model = MPNNModel(mode='classification', n_tasks=1,
  ...                  batch_size=16, learning_rate=0.001)
  >>> loss =  model.fit(dataset, nb_epoch=5)

  References
  ----------
  .. [1] Fabian B. Fuchs and Daniel E. Worrall and Volker Fischer and Max Welling.
         "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks". NeurIPS 2020

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.
  """

  def __init__(self,
               n_tasks: int,
               node_out_feats: int = 64,
               edge_hidden_feats: int = 128,
               num_step_message_passing: int = 3,
               num_step_set2set: int = 6,
               num_layer_set2set: int = 3,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               self_loop: bool = False,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    node_out_feats: int
      The length of the final node representation vectors. Default to 64.
    edge_hidden_feats: int
      The length of the hidden edge representation vectors. Default to 128.
    num_step_message_passing: int
      The number of rounds of message passing. Default to 3.
    num_step_set2set: int
      The number of set2set steps. Default to 6.
    num_layer_set2set: int
      The number of set2set layers. Default to 3.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    number_bond_features: int
      The length of the initial bond feature vectors. Default to 11.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    self_loop: bool
      Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
      Generally, an MPNNModel does not require self loops. Default to False.
    kwargs
      This can include any keyword argument of TorchModel.
    """
    super(SE3TransformerModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)
        
    model = SE3TransformerInternal(
        n_tasks=n_tasks,
        node_out_feats=node_out_feats,
        edge_hidden_feats=edge_hidden_feats,
        num_step_message_passing=num_step_message_passing,
        num_step_set2set=num_step_set2set,
        num_layer_set2set=num_layer_set2set,
        mode=mode,
        number_atom_features=number_atom_features,
        number_bond_features=number_bond_features,
        n_classes=n_classes)
    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    else:
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']

    self._self_loop = self_loop

  def _prepare_batch(self, batch):
    """Create batch data for MPNN.

    Parameters
    ----------
    batch: tuple
      The tuple is ``(inputs, labels, weights)``.

    Returns
    -------
    inputs: DGLGraph
      DGLGraph for a batch of graphs.
    labels: list of torch.Tensor or None
      The graph labels.
    weights: list of torch.Tensor or None
      The weights for each sample or sample/task pair converted to torch.Tensor.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')

    inputs, labels, weights = batch
    dgl_graphs = [
        graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
    ]
    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(MPNNModel, self)._prepare_batch(([], labels,
                                                                weights))
    return inputs, labels, weights
