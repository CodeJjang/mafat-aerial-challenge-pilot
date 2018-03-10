from datetime import datetime


class Logger:
    @staticmethod
    def log(*args):
        '''
        Print arbitrary length strings as a log with timestamp.
        '''
        args_str = ''
        for arg in args:
            args_str += str(arg) + ' '
        print(Logger._get_time_str() + ' ' + args_str)

    @staticmethod
    def _get_time_str():
        return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
