
class LoggingTimer:
    import time

    def __init__(self):
        self.timings = dict()

        self._starts = dict()

    def start(self, key):

        if key not in self.timings:
            self.timings[key] = list()
        if key not in self._starts:
            self._starts[key] = None

        if self._starts[key] is None:

            self._starts[key] = self._stamp()

    def stop(self, key, note=""):

        if key not in self.timings:
            self.timings[key] = list()
        if key not in self._starts:
            self._starts[key] = None

        if self._starts[key] is not None:

            stop = self._stamp()
            elapsed = self._compute_elapsed(self._starts[key], stop)
            self.timings[key].append((elapsed, note))

            self._starts[key] = None

    def _stamp(self):

        return self.time.perf_counter()

    def _compute_elapsed(self, start, stop):

        return stop - start

    def __str__(self):

        s = ""

        for key, data in self.timings.items():

            for d in data:

                s += f"{key}, {d[0]}, {d[1]}\n"

        return s

    def to_csv(self, fname):

        with open(fname, 'w') as outfile:

            outfile.write(str(self))

class MPILoggingTimer(LoggingTimer):

    from mpi4py import MPI

    def _stamp(self):

        return self.MPI.Wtime()

    def __str__(self):

        s = ""

        for key, data in self.timings.items():

            for d in data:

                s += f"{key}, {self.MPI.COMM_WORLD.rank}, {d[0]}, {d[1]}\n"

        return s


# import numpy as np


# timer = LoggingTimer()

# for k in range(5):
#     for i in range(10, 12):

#         n = 2**i
#         key = f"{n} X {n}"

#         timer.start(key)

#         a = np.ones((n,n))

#         timer.stop(key, k)

# timer.to_csv("test.csv")

# print(f"Sequential:\n------------\n{timer}")


# timer2 = MPILoggingTimer()

# for k in range(5):
#     for i in range(10, 12):

#         n = 2**i
#         key = f"{n} X {n}"

#         timer2.start(key)

#         a = np.ones((n,n))

#         timer2.stop(key, k)

# print(f"Parallel:\n------------\n{timer2}")
