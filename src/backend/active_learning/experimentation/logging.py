import os
import traceback


class Log:
    def __init__(self, log_dir, filename='log'):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, '{}.txt'.format(filename))
        self.log_file = open(self.path, 'w')
        return

    def log(self, *args, sep=' ', end='\n', quite=False, **kwargs):
        """
        writes text both to console output and the log file
        """
        print(*args, sep=sep, end=end, file=self.log_file, flush=True)
        if not quite:
            print(*args, sep=sep, end=end, **kwargs)
        return

    def close(self):
        self.log_file.close()
        return

    def report_log_error(self, error):
        self.l("Error occured!")
        self.l(traceback.format_exc())
        self.close()
        return
