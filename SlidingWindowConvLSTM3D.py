class SlidingWindowConvLSTM3D(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels,
            kernel_size,
            station_dim: int,
            component_dim: int,
            num_layers: int = 1,
            window_size: int = 50,
            stride: int = 10,
            chunk_size: int = 150
    ):
        super(SlidingWindowConvLSTM3D, self).__init__()

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        if isinstance(kernel_size, int) or isinstance(kernel_size[0], int):
            kernel_size = [kernel_size] * num_layers

        assert len(hidden_channels) == num_layers
        assert len(kernel_size) == num_layers

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.station_dim = station_dim
        self.component_dim = component_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = chunk_size

        cell_list = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                ConvLSTM3DCell(
                    input_channels=in_ch,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size[i],
                    station_dim=station_dim,
                    component_dim=component_dim
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def _init_hidden(self, batch_size: int, device: torch.device):
        hidden = []
        for ch in self.hidden_channels:
            h = torch.zeros(batch_size, ch, self.station_dim, self.component_dim, device=device)
            c = torch.zeros_like(h)
            hidden.append((h, c))
        return hidden

    def forward(self, x: torch.Tensor, hidden_state=None):
        """
        x: [B, C, T, S, K]
        return:
            final_output: [B, H_last, T, S, K]
            hidden_state: list of (h, c)
        """
        B, _, T, S, K = x.size()
        device = x.device

        if hidden_state is None:
            hidden_state = self._init_hidden(B, device)

        num_windows = max(1, math.ceil((T - self.window_size) / self.stride) + 1)
        H_last = self.hidden_channels[-1]

        final_output = torch.zeros(B, H_last, T, S, K, device=device)
        count_map = torch.zeros(B, H_last, T, S, K, device=device)

        for w in range(num_windows):
            t0 = w * self.stride
            t1 = min(t0 + self.window_size, T)

            window_input = x[:, :, t0:t1, :, :]
            win_len = window_input.size(2)

            layer_in = window_input
            new_hidden = []

            for i, cell in enumerate(self.cell_list):
                h_prev, c_prev = hidden_state[i]
                num_chunks = max(1, math.ceil(win_len / self.chunk_size))

                chunk_h_outputs = []
                chunk_c_outputs = []

                for chunk_idx in range(num_chunks):
                    ct0 = chunk_idx * self.chunk_size
                    ct1 = min((chunk_idx + 1) * self.chunk_size, win_len)

                    chunk_in = layer_in[:, :, ct0:ct1, :, :]
                    chunk_len = chunk_in.size(2)

                    h_ext = h_prev.unsqueeze(2).expand(-1, -1, chunk_len, -1, -1)
                    c_ext = c_prev.unsqueeze(2).expand(-1, -1, chunk_len, -1, -1)

                    h_out, c_out = cell(chunk_in, h_ext, c_ext)

                    chunk_h_outputs.append(h_out)
                    chunk_c_outputs.append(c_out)

                    h_prev = h_out[:, :, -1, :, :].contiguous()
                    c_prev = c_out[:, :, -1, :, :].contiguous()

                layer_out = torch.cat(chunk_h_outputs, dim=2)
                cell_out = torch.cat(chunk_c_outputs, dim=2)

                state_idx = min(self.stride - 1, win_len - 1)
                new_hidden.append((
                    layer_out[:, :, state_idx, :, :].contiguous(),
                    cell_out[:, :, state_idx, :, :].contiguous()
                ))

                layer_in = layer_out

            hidden_state = new_hidden

            final_output[:, :, t0:t1] += layer_in
            count_map[:, :, t0:t1] += 1.0

        final_output = final_output / torch.clamp(count_map, min=1.0)
        return final_output, hidden_state