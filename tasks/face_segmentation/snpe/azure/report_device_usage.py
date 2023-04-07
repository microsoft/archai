# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import dateutil.parser
import datetime
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def parse_date(date):
    s = f"{date}".strip()
    date = dateutil.parser.isoparse(s)
    date = date.replace(tzinfo=datetime.timezone.utc)
    return date


def get_usage_by_device(store: ArchaiStore, report_start, report_end):
    devices = {}

    first = None
    last = None

    for e in store.get_all_status_entities():
        device = e['name']
        start = parse_date(e['start'])
        end = parse_date(e['end'])
        if report_start is not None and report_start > start:
            continue
        if report_end is not None and report_end < start:
            continue
        if report_end is not None and end > report_end:
            end = report_end
        if device not in devices:
            devices[device] = []
        devices[device] += [(start, end)]
        if first is None or start < first:
            first = start
        if last is None or end > last:
            last = end

    return (devices, first, last)


def report(store: ArchaiStore, report_start, report_end):
    devices, first, last = get_usage_by_device(store, report_start, report_end)
    if first is None:
        print("No data found")
        return

    # column headings
    print("date,{}".format(",".join([k for k in devices])))

    start = datetime.datetime(first.year, first.month, first.day, 0, 0, 0, 0, first.tzinfo)
    last = datetime.datetime(last.year, last.month, last.day, 23, 59, 59, 999999, first.tzinfo)
    while start < last:
        du = []
        end = start + datetime.timedelta(days=1)
        total = (end - start).total_seconds()
        for k in devices:
            s = devices[k]
            used = 0
            for d in s:
                ds = d[0]
                de = d[1]
                if ds > end or de < start:
                    continue
                if ds < start:
                    ds = start
                if de > end:
                    de = end
                u = (de - ds).total_seconds()
                if u < 0:
                    print("?")
                used += u

            x = int((used * 100) / total)
            du += [x]

        st = start.strftime("%x")
        print("{},{}".format(st, ",".join([str(x) for x in du])))
        start = end

    total_seconds = (last - first).total_seconds()
    total_used = []
    for k in devices:
        s = devices[k]
        used = 0
        for d in s:
            u = (d[1] - d[0]).total_seconds()
            used += u

        x = int((used * 100) / total_seconds)
        total_used += [x]

    print("total,{}".format(",".join([str(x) for x in total_used])))


if __name__ == '__main__':
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Report on Qualcomm device utilization in an optional date range. ' +
                    'Reports percentage utilization per day.')
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

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key, table_name='usage')
    report(store, start, end)
