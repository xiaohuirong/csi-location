import numpy as np
from utils.parse_args import parse_args, show_args
import tqdm
import multiprocessing as mp


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over


data_part_path = f"data/round{r}/s{s}/data/Port{p}Over{o}Round{r}InputData{s}.npy"
adp_path = f"data/round{r}/s{s}/feature/Port{p}Over{o}Adp{r}Scene{s}.npy"

# (20000, 32, 100)
H = np.load(data_part_path, mmap_mode="r")
[bsz, ant_num, sr_num] = H.shape

adp_dissimilarity_matrix = np.zeros((bsz, bsz), dtype=np.float32)


def adp_dissimilarities_worker(todo_queue, output_queue):
    def adp_dissimilarities(index):
        h = H[index, :, :]
        w = H[index:, :, :]

        dotproducts = (
            np.abs(np.einsum("at,lat->lt", np.conj(h), w, optimize="optimal")) ** 2
        )
        norms = np.real(
            np.einsum("at,at->t", h, np.conj(h), optimize="optimal")
            * np.einsum("lat,lat->lt", w, np.conj(w), optimize="optimal")
        )

        return np.sum(1 - dotproducts / norms, axis=(-1))

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
