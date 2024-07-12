import numpy as np
from utils.parse_args import parse_args, show_args
import tqdm
import multiprocessing as mp


args = parse_args()
show_args(args)

r = args.round
s = args.scene

data_path = f"data/round{r}/s{s}/data/Round{r}InputData{s}_S.npy"
dis_f_path = f"data/round{r}/s{s}/feature/FDis{r}Scene{s}_S.npy"

H = np.load(data_path, mmap_mode="r")
[bsz, port_num, ant_num, sr_num] = H.shape
H = H.reshape(bsz, port_num * 2, ant_num // 2, sr_num)

# H (bsz, 2, 2, 32, 100)
H = np.fft.ifft(H)[..., 0:100]


adp_dissimilarity_matrix = np.zeros((bsz, bsz, 3, 4), dtype=np.float32)


def adp_dissimilarities_worker(todo_queue, output_queue):
    def adp_dissimilarities(index):
        h = H[index, :, :, :]
        w = H[index:, :, :, :]

        dotproducts = (
            np.abs(np.einsum("pat,lpat->lpt", np.conj(h), w, optimize="optimal")) ** 2
        )
        norms = np.real(
            np.einsum("pat,pat->pt", h, np.conj(h), optimize="optimal")
            * np.einsum("lpat,lpat->lpt", w, np.conj(w), optimize="optimal")
        )

        return [
            np.mean(1 - dotproducts / norms, axis=-1),
            np.mean(dotproducts, axis=-1),
            np.mean(norms, axis=-1),
        ]

    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        output_queue.put((index, adp_dissimilarities(index)))


with tqdm.tqdm(total=bsz**2) as pbar:
    todo_queue = mp.Queue()
    output_queue = mp.Queue()

    for i in range(H.shape[0]):
        todo_queue.put(i)

    thread_num = 12
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
            for k in range(3):
                adp_dissimilarity_matrix[i, i:, k] = d[k]
                adp_dissimilarity_matrix[i:, i, k] = d[k]
            pbar.update(2 * len(d[0]) - 1)

np.save(dis_f_path, adp_dissimilarity_matrix)
