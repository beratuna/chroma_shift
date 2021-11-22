from einops import rearrange, reduce, repeat

import torch, torchvision

class Flow(torch.nn.Module):
    def __init__(self, height, width, rand_param=1, parameterization=None, init_method="normal"):

        # parameterization is the function to apply
        # to self.flow_field before applying the flow

        super().__init__()

        self.H = height
        self.W = width
        
        if parameterization == None:
            self.parameterization = torch.nn.Identity()
        else:
            self.parameterization = parameterization

        self.pre_flow_field = torch.nn.Parameter(
            torch.randn([2, self.H, self.W]) * rand_param, requires_grad=True
        )

    def forward(self, x):
        assert (
            self.H == x.shape[-2] and self.W == x.shape[-1]
        ), "flow is initialized with different shape than image"

        BATCH_SIZE = x.shape[0]
        applied_flow_field = self.parameterization(self.pre_flow_field)

        grid = torch.stack(
            torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        ).to(self.pre_flow_field.device)
        # grid: Tensor[2, H, W] // grid[:, n, m] = (n, m)

        batched_grid = repeat(
            grid,
            "c h w -> b h w c",
            b=BATCH_SIZE,
        )

        applied_flow_field_batch = repeat(
            applied_flow_field, "yx h w -> b h w yx", b=BATCH_SIZE
        )
        sampling_grid = batched_grid + applied_flow_field_batch
        return self.sample_grid(x, sampling_grid)

    def sample_grid(self, img_batch, grid_batch):
        """
        Samples img_batch with indices in grid_batch applying bilinear interpolation
        """
        num_channels = img_batch.shape[1]

        added = repeat(
            torch.tensor(
                [[0, 0], [0, 1], [1, 0], [1, 1]],
                dtype=torch.long,
                device=img_batch.device,
            ),
            "ind c -> b h w ind c",
            b=img_batch.shape[0],
            h=img_batch.shape[-2],
            w=img_batch.shape[-1],
        )

        sampled_pixel_coordinates = torch.add(
            repeat(torch.floor(grid_batch).long(), "b h w c -> b h w 4 c"), added
        )

        sampled_pixel_distances = torch.abs(
            torch.sub(
                sampled_pixel_coordinates,
                repeat(grid_batch, "b h w c -> b h w c2 c", c2=4),
            )
        )

        sampled_pixel_distances_h = sampled_pixel_distances[:, :, :, :, 0]
        sampled_pixel_distances_w = sampled_pixel_distances[:, :, :, :, 1]
        sampled_pixel_weights = (1 - sampled_pixel_distances_h) * (
            1 - sampled_pixel_distances_w
        )

        sampled_pixel_coordinates_y = sampled_pixel_coordinates[:, :, :, :, 0]
        sampled_pixel_coordinates_x = sampled_pixel_coordinates[:, :, :, :, 1]

        sampled_pixel_coordinates_y = torch.clamp(
            sampled_pixel_coordinates_y, 0, self.H - 1
        )
        sampled_pixel_coordinates_x = torch.clamp(
            sampled_pixel_coordinates_x, 0, self.W - 1
        )

        sampled_pixel_indices = (
            sampled_pixel_coordinates_y * self.W + sampled_pixel_coordinates_x
        )

        sampled_pixel_indices_flat = repeat(
            sampled_pixel_indices,
            "b h w four -> b c (h w four)",
            c=num_channels,
        )

        img_batch_flat = rearrange(img_batch, "b c h w -> b c (h w)")

        sampled_pixels_flat = torch.gather(
            input=img_batch_flat, index=sampled_pixel_indices_flat, dim=-1
        )

        sampled_pixels = rearrange(
            sampled_pixels_flat,
            "b c (h w four) -> b c h w four",
            h=self.H,
            w=self.W,
            four=4,
        )

        sampled_pixels_weighted = sampled_pixels * repeat(
            sampled_pixel_weights,
            "b h w four -> b c h w four",
            c=num_channels,
        )
        sampled_pixels_weighted_sum = reduce(
            sampled_pixels_weighted, "b c h w four -> b c h w", reduction="sum"
        )

        return sampled_pixels_weighted_sum

    def get_applied_flow(self):
        return self.parameterization(self.pre_flow_field)


def flow_loss(flow_layer):
    applied_flow = flow_layer.get_applied_flow()
    padded_flow = torchvision.transforms.Pad(
        padding=[1, 1],
        padding_mode="reflect",
    )(applied_flow)

    right = padded_flow[:, 1:-1, 2:]
    left = padded_flow[:, 1:-1, :-2]
    down = padded_flow[:, 2:, 1:-1]
    up = padded_flow[:, :-2, 1:-1]

    d_right = torch.sqrt(torch.sum(torch.square(applied_flow - right), dim=0))
    d_left = torch.sqrt(torch.sum(torch.square(applied_flow - left), dim=0))
    d_down = torch.sqrt(torch.sum(torch.square(applied_flow - down), dim=0))
    d_up = torch.sqrt(torch.sum(torch.square(applied_flow - up), dim=0))
    
    total = torch.sum(d_right + d_left + d_down + d_up)

#     if(self.PN == 8):
#         up_right = padded_flow[:, :-2, 2:]
#         up_left = padded_flow[:, 2:, :-2]
#         down_right = padded_flow[:, 2:, 2:]
#         down_left = padded_flow[:, 2:, :-2]

#         d_up_right = torch.sqrt(torch.sum(torch.square(applied_flow - up_right), dim=0))
#         d_up_left = torch.sqrt(torch.sum(torch.square(applied_flow - up_left), dim=0))
#         d_down_right = torch.sqrt(torch.sum(torch.square(applied_flow - down_right), dim=0))
#         d_down_left = torch.sqrt(torch.sum(torch.square(applied_flow - down_left), dim=0))
#         total += torch.sum(d_up_right + d_down_right + d_down_left)

    return total


if __name__ == "__main__":
    img = torch.randn(1, 3, 100, 224)
    flow = Flow(height=100, width=224, parameterization=lambda x: torch.tanh(x))

    flowed_img = flow(img)
    print(flow_loss(flow))
