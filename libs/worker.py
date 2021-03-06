import Queue
from threading import Thread

class BaseWorker
    """ Entry point for process.
    """
    def start(self, action_in_q, data_out_q, **kwargs):
        self.local_data_q = Queue.Queue()
        self.action_in_q = action_in_q
        self.data_out_q = data_out_q

        # Start sending thread
        t = Thread(target=self.send_data_back, args=(data_out_q, local_data_q))
        t.daemon = True
        t.start()

        self.run(**kwargs)

    def send_data_back(self):
        while True:
            o = self.local_data_q.get(block=True)
            self.data_out_q.put(o, block=True)

    def enq_out_data(self, o):
        self.local_data_q.put(o)

    def has_pending_action(self):
        return not self.action_in_q.empty()

    def get_pending_action(self):
        return self.action_in_q.get()

    def run(self, **kwargs):
        raise NotImplementedError
