import torch


def compute_locations(features, stride):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.device
    )
    return locations_per_level


# stride = 8
def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    # len(shifts_x) = 25
    # 返回所有方格的两个x, y坐标矩阵, 共有x.shape[0] * x.shape[1]个方格
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
    # print('shift_x:\n', shift_x)
    # print('shift_y:\n', shift_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # print('shift_x:\n', shift_x)
    # print('shift_y:\n', shift_y)

    # locations = torch.stack((shift_x, shift_y), dim=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride
    # [0,...,192] + 32 = [32,...,224]
    locations = torch.stack((shift_x, shift_y), dim=1) + 32  # alex:48 // 32
    return locations


if __name__ == '__main__':
    loc = compute_locations_per_level(25, 25, 8, 'cuda')
    # loc = compute_locations_per_level(6, 4, 2, 'cuda')
    print(loc)
