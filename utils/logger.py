import sys
import logging
from functools import wraps

class Logger(object):

    def __init__(self, filename=''):

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        if filename:
            fh = logging.FileHandler(filename)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def concat_message(f):
        @wraps(f)
        def decorated(self, *args):
            mess =  ' '.join(args)
            f(self, mess)
        return decorated

    @concat_message
    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    @concat_message
    def info(self, message):
        self.logger.info(message)
        self._flush()

    @concat_message
    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    @concat_message
    def error(self, message):
        self.logger.error(message)
        self._flush()

    @concat_message
    def critical(self, message):
        self.logger.critical(message)
        self._flush()


if __name__ == '__main__':
    log = Logger('test.log')
    log.debug('debug')
    log.info('info', 'test')
    log.warning('warning')
    log.error('error')
    log.critical('critical')
