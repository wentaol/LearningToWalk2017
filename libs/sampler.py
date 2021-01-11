from multiprocessing import Process
import multiprocessing
import Queue
from threading import Thread

class Sampler(object):
    def __init__(self, n_workers=1):
        self.n_workers = n_workers
        self.collected_data_q = Queue.Queue()
        self.actions_q = Queue.Queue()
        self.worker_action_queues = []

    def start_workers(self, worker, worker_args=None):
        """ Start the pool of processes. They do nothing until they receive a model.
        """
        if not worker_args:
            worker_args = dict()

        for i in range(self.n_workers):
            worker_to_master_q = multiprocessing.Queue()
            master_to_worker_q = multiprocessing.Queue() 
            self.worker_action_queues.append(master_to_worker_q)


            p = Process(target=worker.start, args=(master_to_worker_q, worker_to_master_q, worker_args))
            p.start()

            t = Thread(target=self._receive_data, args=(worker_to_master_q, ))
            t.daemon = True
            t.start()

        t = Thread(target=self._send_actions)
        t.daemon = True
        t.start()

    def collect_data(self, block=False):
        """ Return data packets collected from workers.
            Blocks until at least one packet of data is available.
        """
        collected_data_packets = []
        try:
            data_packet = self.collected_data_q.get(block=block)
            collected_data_packets.append(data_packet)
        except Queue.Empty:
            pass
        while True:
            try:
                data_packet = self.collected_data_q.get(block=False)
                collected_data_packets.append(data_packet)
            except Queue.Empty:
                break

        return collected_data_packets

    def update_agent(self, agent):
        """ Propagates a new agent to all workers.
            Async, so this function returns before the update is complete.
        """
        action = 'update_agent'
        args = dict(
            weights=agent.get_weights()
            )
        self.actions_q.put((action, args))

    def _receive_data(self, data_in_q):
        while True:
            e = data_in_q.get(block=True)
            self.collected_data_q.put(e, block=False)

    def _send_actions(self):
        while True:
            a = self.actions_q.get(block=True)
            for q in self.worker_action_queues:
                q.put_nowait(a)
