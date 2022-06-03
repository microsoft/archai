import argparse
import dateutil.parser
import datetime
from usage import get_all_usage_entities


def parse_date(date):
    s = f"{date}".strip()
    date = dateutil.parser.isoparse(s)
    date = date.replace(tzinfo=datetime.timezone.utc)
    return date


def report(report_start, report_end):
    devices = {}
    for e in get_all_usage_entities():
        device = e['name']
        start = parse_date(e['start'])
        end = parse_date(e['end'])
        if report_start is not None and report_start > start:
            continue
        if report_end is not None and report_end < end:
            end = report_end
        if device not in devices:
            devices[device] = []
        devices[device] += [(start, end)]

    for k in devices:
        data = devices[k]
        s = sorted(data, key=lambda x: x[0])
        start = s[0][0]
        end = s[-1][1]
        total = (end - start).total_seconds()
        used = 0
        for d in s:
            u = (d[1] - d[0]).total_seconds()
            used += u

        x = int((used * 100) / total)
        st = start.strftime("%x %X")
        et = end.strftime("%x %X")
        print(f"Device {k} used from {st} to {et} was utilized {x} percent")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Report on Qualcomm device utilization from an optional start date.')
    parser.add_argument('--start', help='Set the "start" date to start the search. (default None).')
    parser.add_argument('--end', help='Set the "end" date to end the search. (default None).')
    args = parser.parse_args()
    start = None
    end = None
    if args.start:
        start = dateutil.parser.parse(args.start)
        start = start.replace(tzinfo=datetime.timezone.utc)
    if args.end:
        end = dateutil.parser.parse(args.end)
        end = end.replace(tzinfo=datetime.timezone.utc)

    report(start, end)
