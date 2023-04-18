import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from experiments.darts_utils.cell_operations import ResNetBasicblock
from experiments.darts_utils.search_cells import NAS201SearchCell_PartialChannel as SearchCell
from experiments.darts_utils.genotypes import Structure
from experiments.darts_utils.utils import process_step_matrix, prune
import logging
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

class TinyNetwork(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, affine=False, track_running_stats=True, k=2, species='softmax', reg_type='l2', reg_scale=1e-3):
    super(TinyNetwork, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self._criterion = criterion
    self.k = k
    self.species = species
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))

    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for _, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats, k)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self._arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )
    self.tau = 10 if species == 'gumbel' else None
    self._mask = None

    #### reg
    self.reg_type = reg_type
    self.reg_scale = reg_scale
    self.anchor = Dirichlet(torch.ones_like(self._arch_parameters).cuda())

  def _loss(self, input, target):
    logits = self(input)
    loss = self._criterion(logits, target)
    if self.reg_type == 'kl':
      loss += self._get_kl_reg()
    return loss

  def _get_kl_reg(self):
    assert(self.species == 'dirichlet') # kl implemented only for Dirichlet
    cons = (F.elu(self._arch_parameters) + 1)
    q = Dirichlet(cons)
    p = self.anchor
    kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
    return kl_reg

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def set_tau(self, tau):
    self.tau = tau

  def get_tau(self):
    return self.tau

  def arch_parameters(self):
    return [self._arch_parameters]

  def show_arch_parameters(self):
    with torch.no_grad():
      logging.info('arch-parameters :\n{:}'.format(process_step_matrix(self._arch_parameters, 'softmax', self._mask).cpu()))
      if self.species == 'dirichlet':
        logging.info('concentration :\n{:}'.format((F.elu(self._arch_parameters) + 1).cpu()))

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    alphas = process_step_matrix(self._arch_parameters, 'softmax', self._mask)
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = alphas[ self.edge2index[node_str] ]
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def pruning(self, num_keep):
    self._mask = prune(self._arch_parameters, num_keep, self._mask)

  def forward(self, inputs):
    alphas = process_step_matrix(self._arch_parameters, self.species, self._mask, self.tau)

    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell(feature, alphas)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)
    return logits

  def wider(self, k):
    self.k = k
    for cell in self.cells:
      if isinstance(cell, SearchCell):
        cell.wider(k)
