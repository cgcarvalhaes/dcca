import sys


def progress(cnt, cnt_max, length=50):
    done = int(round(length * cnt / float(cnt_max)))
    done_percent = 100. * cnt / cnt_max
    sys.stdout.write('%s %.1f%%\r' % ('#' * done, done_percent))
    sys.stdout.flush()
   