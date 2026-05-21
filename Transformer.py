import torch
import torch.nn as nn
import torch.optim as optim

class InductionHeadTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 4
        self.d_model = 6
        self.n_heads = 2
        self.n_layers = 2

        self.seq_len = 10 

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.seq_len, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        
    def forward(self, x):
        seq_length = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

def train_and_generate():
    torch.manual_seed(42)
    model = InductionHeadTransformer()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(1500):
        half = torch.randint(0, 4, (128, 5))
        seq = torch.cat([half, half], dim=1)
        
        x = seq[:, :-1]
        y = seq[:, 1:]
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, 4), y.reshape(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    chars = ['A', 'B', 'C', 'D']

    input_seq = [0, 1, 0, 2, 3]
    context = torch.tensor([input_seq])
    
    print(" ".join(chars[idx] for idx in input_seq), "|", end="", flush=True)

    with torch.no_grad():
        for _ in range(5):
            logits = model(context)
            next_token_id = logits[0, -1, :].argmax().item()
            
            print(" " + chars[next_token_id], end="", flush=True)
            
            context = torch.cat([context, torch.tensor([[next_token_id]])], dim=1)
    
    print()

if __name__ == "__main__":
    train_and_generate()