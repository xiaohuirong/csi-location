import numpy as np
from utils.parse_args import parse_args, show_args
import matplotlib.pyplot as plt
import tqdm
import multiprocessing as mp

args = parse_args()
show_args(args)

# h (bsz, port_num, ant_num, sc_num) = (2000, 2, 64, 408)
csi_time_domain = np.load("data/Round0InputData1_S1_new.npy", mmap_mode="r")

csi_time_domain = np.fft.ifft(csi_time_domain)

# abs_csi_time_domain = np.abs(csi_time_domain)
# # (bsz, sc_num)
# mean_csi_time_domain = np.mean(abs_csi_time_domain, axis=(1, 2))
# max_index = np.argmax(mean_csi_time_domain, axis=1)

bsz = csi_time_domain.shape[0]
csi_time_domain = csi_time_domain.reshape(bsz, 2, 2, 8, 4, 408)

adp_dissimilarity_matrix = np.zeros((bsz, bsz), dtype=np.float32)


def adp_dissimilarities_worker(todo_queue, output_queue):
    def adp_dissimilarities(index):
        # h has shape (arrays, antennas, taps), w has shape (datapoints, arrays, antennas, taps)
        h = csi_time_domain[index, :, :, :, :]
        w = csi_time_domain[index:, :, :, :, :]

        dotproducts = (
            np.abs(np.einsum("poart,lpoart->lport", np.conj(h), w, optimize="optimal"))
            ** 2
        )
        norms = np.real(
            np.einsum("poart,poart->port", h, np.conj(h), optimize="optimal")
            * np.einsum("lpoart,lpoart->lport", w, np.conj(w), optimize="optimal")
        )

        return np.sum(1 - dotproducts / norms, axis=(1, 2, 3, 4))

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

    thread_num = 8
    for i in range(thread_num):
        todo_queue.put(-1)
        p = mp.Process(
            target=adp_dissimilarities_worker, args=(todo_queue, output_queue)
        )
        p.start()

    finished_processes = 0
    while finished_processes != thread_num:
        i, d = output_queue.get()

        if i == -1:
            finished_processes = finished_processes + 1
        else:
            adp_dissimilarity_matrix[i, i:] = d
            adp_dissimilarity_matrix[i:, i] = d
            pbar.update(2 * len(d) - 1)

np.save("data/Round0InputData1_S1_ADP.npy", adp_dissimilarity_matrix)
