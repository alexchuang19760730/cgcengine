import sys

p = "model.py"
s = open(p).read()

# add module-global REF_HS before Transformer class
s = s.replace("class Transformer(nn.Module):", "REF_HS = []\n\n\nclass Transformer(nn.Module):", 1)

old = '''    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits'''

new = '''    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        global REF_HS
        if start_pos == 0:
            REF_HS = []
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
            if start_pos == 0:
                REF_HS.append(h[:, -1].detach().float().cpu().clone())
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        if start_pos == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            torch.save(REF_HS, "/data/ref_hs.pt")
        return logits'''

assert old in s, "OLD BLOCK NOT FOUND"
s = s.replace(old, new)
open(p, "w").write(s)
print("patched OK")
