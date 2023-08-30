worker_class = "uvicorn.workers.UvicornWorker"

# Number of worker process
workers = 1
# Number of threads each worker process,
# (workers * threads) means how many requests the app can process simultaneously
threads = 32
# Number of requests that can be waiting to be served,
# requests exceed this number will be rejected and receive an error
backlog = 64
# Workers silent for more than this many seconds are killed and restarted
timeout = 300

# log format for access log, error log can't set format
access_log_format = '%(h)s %(l)s %(t)s "%(r)s" %(m)s %(s)s %(b)s "%(f)s" "%(a)s"'

"""
Below are description for each format option:
h          remote address
l          '-'
u          currently '-', may be user name in future releases
t          date of the request
r          status line (e.g. ``GET / HTTP/1.1``)
s          status
b          response length or '-'
f          referer
a          user agent
T          request time in seconds
D          request time in microseconds
L          request time in decimal seconds
p          process ID
"""
