import numpy as np
from utils.parse_args import parse_args, show_args
import matplotlib.pyplot as plt
import tqdm
import multiprocessing as mp

args = parse_args()
show_args(args)

# h (bsz, port_num, ant_num, sc_num) = (2000, 2, 64, 408)
csi_time_domain = np.load("data/Round0InputData1_S1.npy", mmap_mode="r")

# (2000, 2, 64, 10)
csi_time_domain = csi_time_domain[..., 0:10]

adp_dissimilarity_matrix = np.zeros(
    (csi_time_domain.shape[0], csi_time_domain.shape[0]), dtype=np.float32
)


def adp_dissimilarities_worker(todo_queue, output_queue):
    def adp_dissimilarities(index):
        # h has shape (arrays, antennas, taps), w has shape (datapoints, arrays, antennas, taps)
        h = csi_time_domain[index, :, :, :]
        w = csi_time_domain[index:, :, :, :]

        dotproducts = (
            np.abs(np.einsum("bmt,lbmt->lbt", np.conj(h), w, optimize="optimal")) ** 2
        )
        norms = np.real(
            np.einsum("bmt,bmt->bt", h, np.conj(h), optimize="optimal")
            * np.einsum("lbmt,lbmt->lbt", w, np.conj(w), optimize="optimal")
        )

        return np.sum(1 - dotproducts / norms, axis=(1, 2))

    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        output_queue.put((index, adp_dissimilarities(index)))


with tqdm.tqdm(total=csi_time_domain.shape[0] ** 2) as pbar:
    todo_queue = mp.Queue()
    output_queue = mp.Queue()

    for i in range(csi_time_domain.shape[0]):
        todo_queue.put(i)

    for i in range(mp.cpu_count()):
        todo_queue.put(-1)
        p = mp.Process(
            target=adp_dissimilarities_worker, args=(todo_queue, output_queue)
        )
        p.start()

    finished_processes = 0
    while finished_processes != mp.cpu_count():
        i, d = output_queue.get()

        if i == -1:
            finished_processes = finished_processes + 1
        else:
            adp_dissimilarity_matrix[i, i:] = d
            adp_dissimilarity_matrix[i:, i] = d
            pbar.update(2 * len(d) - 1)

np.save("data/Round0InputData1_S1_ADP.npy", adp_dissimilarity_matrix)
