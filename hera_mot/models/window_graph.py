"""Strong detector tokens -> real four-frame path cover -> path descriptor.

Window and endpoint controls share parameter shapes and exact path decoding.
Endpoint scores are computed on isolated two-node graphs, batched over edges.
"""
import numpy as np
import torch
from torch import nn

from .relation_encoder import ConnectionRefinement, RefinementConfig, ConnectionRound
from ..tracker.graph import decode_graph


class StrongWindowGraph(ConnectionRefinement):
    def __init__(self, appearance_dim=256):
        super().__init__(RefinementConfig(appearance_dim=int(appearance_dim)))

    def forward(self, window, graph, context='window'):
        if context == 'window':
            return super().forward(window, graph, feedback=False)
        if context == 'window_feedback':
            return super().forward(window, graph, feedback=True)
        if context == 'competition':
            return super().forward(
                window, graph, feedback=False, shared_context=False
            )
        if context not in ('pair', 'point_private', 'shared_private'):
            raise ValueError(
                'context must be pair, point_private, shared_private, competition, window, or window_feedback'
            )
        if window.window_size > 4 or not np.array_equal(graph.frames, window.frames):
            raise ValueError('Invalid causal window')
        if not len(graph.edges):
            zero = self.edge_score.weight.sum().expand(0)
            return [ConnectionRound(zero,decode_graph(graph,np.empty(0)),np.empty((0,4),np.int64)) for _ in range(self.config.rounds)]
        device = next(self.parameters()).device
        endpoints = torch.as_tensor(graph.edges,device=device)
        all_boxes = torch.as_tensor(window.bboxes,dtype=torch.float32,device=device)
        all_scores = torch.as_tensor(window.confidences,dtype=all_boxes.dtype,device=device)
        all_features = torch.as_tensor(window.appearance,dtype=all_boxes.dtype,device=device)
        scale = all_boxes.new_tensor([window.image_width,window.image_height]*2)
        base_nodes = self.detector(torch.cat((all_features,all_boxes/scale,all_scores[:,None]),-1))
        boxes = all_boxes[endpoints]
        frames = torch.as_tensor(window.frames,device=device)[endpoints]
        if context == 'pair':
            nodes = base_nodes[endpoints]
            nodes = self.context(
                (nodes+self.time(frames-window.start_frame)).transpose(0,1)
            ).transpose(0,1)
        elif context == 'point_private':
            offsets = torch.as_tensor(
                window.frames-window.start_frame, device=device
            )
            pointwise = self.context(
                (base_nodes+self.time(offsets))[None]
            ).squeeze(0)
            nodes = pointwise[endpoints]
        else:
            # Share information across every detection in the window, then
            # keep recurrent messages private to each candidate edge.  This
            # isolates shared node context from cross-candidate competition.
            offsets = torch.as_tensor(
                window.frames-window.start_frame, device=device
            )
            shared = self.context((base_nodes+self.time(offsets))[:,None]).squeeze(1)
            nodes = shared[endpoints]
        edge_features = torch.as_tensor(graph.features,dtype=nodes.dtype,device=device)
        edges = self.edge_input(torch.cat((nodes[:,0],nodes[:,1],edge_features),-1))

        def gains():
            return self.edge_score(edges).squeeze(-1)-self.no_out(nodes[:,0]).squeeze(-1)-self.no_in(nodes[:,1]).squeeze(-1)

        def compete(g):
            # A single candidate edge + null at each endpoint; no other edges.
            message = g.sigmoid()[:,None]*edges
            zeros = torch.zeros_like(message)
            return torch.stack((self.node_update(torch.cat((zeros,message),-1),nodes[:,0]),
                                self.node_update(torch.cat((message,zeros),-1),nodes[:,1])),1)

        nodes = compete(gains())
        rounds=[]
        indices=np.full((len(graph.edges),4),-1,np.int64)
        for r in range(self.config.rounds):
            g=gains();rounds.append(ConnectionRound(g,decode_graph(graph,g.detach().cpu().numpy()),indices.copy()))
            if r+1==self.config.rounds:break
            dt=(frames[:,1]-frames[:,0]).to(nodes.dtype)
            centers=boxes[:,:,:2]+boxes[:,:,2:]/2
            height=((boxes[:,0,3]+boxes[:,1,3])/2).clamp_min(1)
            velocity=(centers[:,1]-centers[:,0])/(height*dt)[:,None]
            size=(boxes[:,1,2:]/boxes[:,0,2:]).log()/dt[:,None]
            motion=nodes.new_zeros((len(edges),2,6))
            motion[:,1]=torch.cat((velocity,size,(dt/3)[:,None],(dt>1).to(nodes.dtype)[:,None]),-1)
            path=self.path_encoder((nodes+self.motion(motion)).transpose(0,1)).mean(0)
            updated_nodes=compete(g)
            updated_edges=self.edge_update(torch.cat((nodes[:,0],nodes[:,1],path,g.new_zeros((len(g),1))),-1),edges)
            nodes,edges=updated_nodes,updated_edges
            indices[:,:2]=graph.edges
        return rounds
