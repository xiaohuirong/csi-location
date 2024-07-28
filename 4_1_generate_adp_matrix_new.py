import numpy as np
from utils.parse_args import parse_args, show_args
import tqdm
import multiprocessing as mp
from scipy.stats import mode


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method


h_path = f"data/round{r}/s{s}/feature/Round{r}Time{s}.npy"
adp_path = f"data/round{r}/s{s}/feature/Adp{r}Scene{s}.npy"

# (20000, 2, 64)
H = np.load(h_path)
[bsz, port_num, ant_num] = H.shape

H = H.reshape(20000, 2, 2, 32)

adp_dissimilarity_matrix = np.zeros((bsz, bsz), dtype=np.float32)


def adp_dissimilarities_worker(todo_queue, output_queue):
    def adp_dissimilarities(index):
        h = H[index, :, :, :]
        w = H[index:, :, :, :]

        dotproducts = (
            np.abs(np.einsum("poa,lpoa->lpo", np.conj(h), w, optimize="optimal")) ** 2
        )
        norms = np.real(
            np.einsum("poa,poa->po", h, np.conj(h), optimize="optimal")
            * np.einsum("lpoa,lpoa->lpo", w, np.conj(w), optimize="optimal")
        )

        return np.sum(1 - dotproducts / norms, axis=(1, 2))

    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        output_queue.put((index, adp_dissimilarities(index)))


with tqdm.tqdm(total=H.shape[0] ** 2) as pbar:
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
            adp_dissimilarity_matrix[i, i:] = d
            adp_dissimilarity_matrix[i:, i] = d
            pbar.update(2 * len(d) - 1)

np.save(adp_path, adp_dissimilarity_matrix)
