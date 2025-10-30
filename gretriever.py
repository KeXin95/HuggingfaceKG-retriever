import torch
from torch_geometric.nn import GAT
from torch_geometric.llm.models import LLM, GRetriever
from torch_geometric.data import Data
import warnings

import torch_geometric.data.data 

warnings.filterwarnings("ignore", category=UserWarning, message=".*Using_kex_impl=False*")

BGE_EMBED_DIM = 768 
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
GNN_HIDDEN_DIM = 256
GNN_OUT_DIM = 256

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = GAT(in_channels, hidden_channels, num_layers=1,
                                   edge_dim=edge_dim)
        self.conv2 = GAT(hidden_channels, out_channels, num_layers=1,
                                   edge_dim=edge_dim)
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

def load_graph_data(filepath: str) -> Data:
    """
    Loads the graph data from the .pt file.
    Includes fix for PyTorch 2.6+ UnpicklingError.
    """
    try:
        graph_data = torch.load(filepath, weights_only=False)
        
        if not isinstance(graph_data, Data):
            raise ValueError(f"Loaded file is a {type(graph_data)}, not a Data object. Please adapt.")
        
        graph_data.x = graph_data.x.float()
        if graph_data.edge_attr is not None:
            graph_data.edge_attr = graph_data.edge_attr.float()
        
        print(f"Successfully loaded graph: {graph_data}")
        return graph_data

    except Exception as e:
        print(f"Error loading graph data from {filepath}: {e}")
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch detected device: {device}")

    print(f"Loading LLM: {LLM_MODEL_NAME}...")
    
    llm = LLM(
        model_name=LLM_MODEL_NAME,
        n_gpus=1
    )
    print(f"LLM loaded and placed itself on: {llm.device}")

    print("Initializing GNN (Graph Encoder)...")
    gnn = GraphEncoder(
        in_channels=BGE_EMBED_DIM,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=GNN_OUT_DIM,
        edge_dim=BGE_EMBED_DIM
    )

    print("Initializing G-Retriever... [GNN ENABLED]")
    model = GRetriever(
        llm=llm,
        gnn=gnn, 
        use_lora=False,
        mlp_out_tokens=1
    )
    model.eval()

    print("Loading graph data...")
    graph_data = load_graph_data('graph_data/final_graph_krystal.pt')
    
    if graph_data is None:
        print("Failed to load graph data. Exiting.")
        return

    question = ["What is the name of justin bieber's brother?"]
    x = graph_data.x
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr
    batch = torch.zeros(x.size(0), dtype=torch.long)

    print(f"\nAsking question: '{question[0]}'")
    
    with torch.no_grad():
        answer = model.inference(
            question=question,
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr
        )

    print(f"\nModel's Answer: {answer[0]}")


if __name__ == "__main__":
    main()