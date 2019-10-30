import numpy as np

# class Filter(object):
#     def __init__(self, size=0, delta=0):
#         self.size = size
#         self.delta = delta
#         self.data = np.zeros(size)
#
#     def A(self):
#         return -self.delta
#
#     def B(self):
#         return self.size - self.delta
#
#     def __getitem__(self, idx):
#         if idx < self.A() or idx >= self.B():
#             raise IndexError
#         return self.data[idx + self.delta]

def recursiveSmooth(img, sigma):
    img = img.copy()
    alpha = 2.5 / (np.sqrt(np.pi) * sigma)
    exp = np.exp(alpha)
    expSqr = exp ** 2
    k = (1.0 - exp) ** 2 / (1.0 + 2.0 * alpha * exp - expSqr)
    preMinus = exp * (alpha - 1.0)
    prePlus = exp * (alpha + 1.0)
    H, W = img.shape[:2]

    valS1_y = np.zeros_like(img)
    valS1_y[0, :, :] = (0.5 - k * preMinus) * img[0, :, :]
    valS1_y[1, :, :] = k * (img[1, :, :] + preMinus * img[0, :, :]) \
                       + (2.0 * exp - expSqr) * valS1_y[0, :, :]
    for y in range(2, H):
        valS1_y[y, :, :] = k * (img[y, :, :] + preMinus * img[y - 1, :, :]) + \
                           2.0 * exp * valS1_y[y - 1, :, :] - expSqr * valS1_y[y - 2, :, :]

    valS2_y = np.zeros_like(img)
    valS2_y[H - 1, :, :] = (0.5 + k * preMinus) * img[H - 1, :, :]
    valS2_y[H - 2, :, :] = k * ((prePlus - expSqr) * img[H - 1, :, :] ) \
                           + (2.0 * exp - expSqr) * valS2_y[H - 1, :, :]
    for y in reversed(range(0, H - 2)):
        valS2_y[y, :, :] = k * (prePlus * img[y + 1, :, :] - expSqr * img[y + 2, :, :]) \
                           + 2.0 * exp * valS2_y[y + 1, :, :] - expSqr[y + 2, :, :]

    img = valS1_y + valS2_y

    valS1_x = np.zeros_like(img)
    valS1_x[:, 0, :] = (0.5 - k * preMinus) * img[:, 0, :]
    valS1_x[:, 1, :] = k * (img[:, 1, :] + preMinus * img[:, 0, :]) \
                       + (2.0 * exp - expSqr) * valS1_x[:, 0, :]
    for x in range(2, W):
        valS1_x[:, x, :] = k * (img[:, x, :] + preMinus * img[:, x - 1, :]) + \
                           2.0 * exp * valS1_x[:, x - 1, :] - expSqr * valS1_x[:, x - 2, :]

    valS2_x = np.zeros_like(img)
    valS2_x[:, H - 1, :] = (0.5 + k * preMinus) * img[:, H - 1, :]
    valS2_x[:, H - 2, :] = k * ((prePlus - expSqr) * img[:, H - 1, :]) \
                           + (2.0 * exp - expSqr) * valS2_x[:, H - 1, :]
    for x in reversed(range(0, W - 2)):
        valS2_x[:, x, :] = k * (prePlus * img[:, x + 1, :] - expSqr * img[:, x + 2, :]) \
                           + 2.0 * exp * valS2_y[:, x + 1, :] - expSqr[:, x + 2, :]

    img = valS1_x + valS2_x
    return img

def flowFilter(flow, ftr=(-0.5, 0, 0.5), ftrdim=0):
    result = np.zeros_like(flow)
    xSize, ySize, _ = flow.shape
    if ftrdim == 0:
        x1, x2 = 0, xSize - 3
        a2Size = 2 * xSize - 1
        for x in range(x1, x2):
            for i in range(len(ftr)):
                result[x] = ftr[i] * flow[x + i]
        for x in range(x2, xSize):
            for i in range(len(ftr)):
                if x + i < 0:
                    result[x] += ftr[i] * flow[-1-x-i]
                elif x + i >= xSize:
                    result[x] += ftr[i] * flow[a2Size - x - i]
                else:
                    result[x] += ftr[i] * flow[x + i]
    else:
        y1, y2 = 0, ySize - 3
        a2Size = 2 * ySize - 1
        for y in range(y1, y2):
            for i in range(len(ftr)):
                result[:,y,:] = ftr[i] * flow[:,y + i,:]
        for y in range(y2, ySize):
            for i in range(len(ftr)):
                if y + i < 0:
                    result[:,y,:] += ftr[i] * flow[:,-1 - y - i,:]
                elif y + i >= xSize:
                    result[:,y,:] += ftr[i] * flow[:,a2Size - y - i,:]
                else:
                    result[:,y,:] += ftr[i] * flow[:,y + i,:]
