"""Build graph edges from committed track endpoints to eligible detections."""
import numpy as np
import torch
from torch import nn


class HistoryNodeGraph(nn.Module):
    def __init__(self, c4, s3, pixel):
        super().__init__()
        self.c4, self.s3, self.pixel = c4, s3, pixel
        self.history_score = nn.Linear(c4.config.dim, 1)
        nn.init.zeros_(self.history_score.weight)
        nn.init.zeros_(self.history_score.bias)
        self.bound = .25

    def history_delta(self, window, graph, history):
        """One node per immutable committed identity endpoint, <=t-1 only.

        The original boundary/Kalman aggregate is retained. Endpoint nodes
        supplement it; legal candidates, HOLD, NEW and ID capacities stay fixed.
        """
        device=next(self.parameters()).device
        h=len(history['ids']);n=graph.size
        edges=np.asarray(history['edges'],np.int64)
        if edges.shape!=(len(edges),2): raise ValueError('history edge shape')
        frames=np.asarray(history['frames'])
        if len(frames)!=h or np.any(frames>=window.start_frame):
            raise ValueError('uncommitted history node')
        if len(set(map(int,history['ids'])))!=h: raise ValueError('duplicate history identity')
        if len(edges) and (edges[:,0].min()<0 or edges[:,0].max()>=h or edges[:,1].min()<0 or edges[:,1].max()>=n):
            raise ValueError('history endpoint outside visible graph')
        if not len(edges): return self.history_score.weight.sum().expand(0)
        boxes=np.concatenate((history['boxes'],window.bboxes))
        scores=np.r_[history['scores'],window.confidences]
        appearance=np.concatenate((history['appearance'],window.appearance))
        x=torch.as_tensor(np.c_[appearance, boxes/np.array([window.image_width,window.image_height]*2),scores],
                          dtype=torch.float32,device=device)
        # Edge features carry the time elapsed since each track endpoint.
        offsets=np.r_[np.zeros(h,np.int64),window.frames-window.start_frame]
        nodes=self.c4.context((self.c4.detector(x)+self.c4.time(torch.as_tensor(offsets,device=device)))[None]).squeeze(0)
        hh=edges+np.array([0,h]);dd=np.asarray(graph.edges)+h
        endpoints=torch.as_tensor(np.concatenate((hh,dd)),device=device)
        feature=torch.as_tensor(np.concatenate((history['features'],graph.features)),dtype=nodes.dtype,device=device)
        u,v=endpoints.T
        emb=self.c4.edge_input(torch.cat((nodes[u],nodes[v],feature),-1))
        for _ in range(self.c4.config.rounds):
            gain=self.c4._gains(nodes,emb,endpoints)
            updated=self.c4._competition(nodes,emb,endpoints,gain)
            # Same recurrent edge operator, explicit two-endpoint context.
            evidence=(nodes[u]+nodes[v])*.5
            emb=self.c4.edge_update(torch.cat((nodes[u],nodes[v],evidence,(gain>0).to(nodes.dtype)[:,None]),-1),emb)
            nodes=updated
        return self.bound*torch.tanh(self.history_score(emb[:len(edges)]).squeeze(-1))

    def pair_delta(self, record, window, graph):
        tokens=torch.as_tensor(record['tokens'],device=next(self.parameters()).device)
        valid=torch.as_tensor(record['valid'],device=tokens.device)
        base=self.pixel(tokens,valid)
        hist=self.history_delta(window,graph,record['history_graph'])
        ix=torch.as_tensor(record['history_pair_indices'],device=tokens.device)
        if len(ix)!=len(base): raise ValueError('path/root alignment')
        return base+torch.cat((hist,hist.new_zeros(1)))[ix]


def history_graph_input(actor, window, graph, anchors):
    """Whitelisted raw measurements from the actual pre-commit ledger."""
    store=actor.store
    ids=np.asarray(anchors.anchor_ids,np.int64)
    indices=store.indices([actor.track_ledger[int(i)][-1] for i in ids])
    frames=store.frames[indices].copy()
    if np.any(frames>=window.start_frame): raise ValueError('history must be committed')
    edges=np.asarray(anchors.edges,np.int64)
    if len(edges):
        a,b=edges.T
        old=store.boxes[indices[a]].astype(np.float64);new=window.bboxes[b].astype(np.float64)
        dt=(window.frames[b]-frames[a]).astype(np.float64)
        height=np.maximum(1.,(old[:,3]+new[:,3])/2)
        velocity=((new[:,:2]+new[:,2:]/2)-(old[:,:2]+old[:,2:]/2))/(height*dt)[:,None]
        size=np.log(new[:,2:]/old[:,2:])/dt[:,None]
        # Compute the history/window cosine table once, not a high-dimensional
        # dot product for every cover or repeated legal-edge occurrence.
        cosines=store.appearance[indices]@window.appearance.T
        features=np.c_[np.minimum(dt,30.)/3,velocity,size,cosines[a,b],anchors.gains].astype(np.float32)
    else:
        features=np.empty((0,7),np.float32)
    return dict(ids=ids,frames=frames,boxes=store.boxes[indices].copy(),scores=store.scores[indices].copy(),
        appearance=store.appearance[indices].copy(),edges=anchors.edges.copy(),
        gains=anchors.gains.copy(),features=np.asarray(features,np.float32).reshape(-1,7))


def history_pair_indices(packet, window, history):
    node_by_key={int(k):i for i,k in enumerate(window.detection_keys)}
    owner_to_index={int(k):i for i,k in enumerate(history['ids'])}
    edge_to_index={tuple(map(int,e)):i for i,e in enumerate(history['edges'])}
    # Every actual OLD path/root is an existing boundary edge. Missing edges
    # retain zero rather than inventing a GT-conditioned candidate.
    return np.asarray([edge_to_index.get((owner_to_index[int(owner)],node_by_key[int(path[0])]),len(history['edges']))
                       for owner,path in zip(packet.owners,packet.paths)],np.int64)
