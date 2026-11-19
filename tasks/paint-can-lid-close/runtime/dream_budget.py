"""Receipts for requests against one episode already reserved in the account ledger."""
import fcntl
import json
import math
import os
from pathlib import Path
import time

class BudgetStopped(RuntimeError):
    pass

class Ledger:
    def __init__(self,path):
        self.path=Path(path)
        try:
            self.receipt=json.loads(self.path.read_text())
            if self.receipt['task']!='paint-can-lid-close' or self.receipt['cap_usd']!=20:
                raise ValueError('Unexpected episode reservation')
            if self.receipt['cumulative_reserved_usd']>self.receipt['cumulative_cap_usd']:
                raise ValueError('Account reservation exceeds authorization')
        except (OSError,ValueError,KeyError) as exc:
            raise BudgetStopped('account_reservation_unavailable_or_invalid') from exc
        self.events=self.path.with_name('request-receipts.jsonl')
        self.lock=self.path.with_name('request-receipts.lock')

    def _read(self):
        if not self.events.exists():
            return []
        try:return [json.loads(line) for line in self.events.read_text().splitlines()]
        except (OSError,ValueError) as exc:raise BudgetStopped('request_receipts_invalid') from exc

    def _append(self,event):
        fd=os.open(self.events,os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600)
        with os.fdopen(fd,'a') as f:
            f.write(json.dumps(event)+'\n')
            f.flush()
            os.fsync(f.fileno())

    def reserve(self,task,run):
        with self.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            rows=self._read()
            reservations=[x for x in rows if x['event']=='reserved']
            if task!=self.receipt['task'] or not run:
                raise BudgetStopped('reservation_task_mismatch')
            if any(x['run']!=run for x in reservations):
                raise BudgetStopped('episode_already_attempted')
            amount=.0015
            count=len(reservations)+1
            if count*amount>self.receipt['cap_usd']:
                raise BudgetStopped('reserved_episode_limit')
            self._append(dict(event='reserved',id=count,run=run,task=task,at=time.time(),
                              estimate_usd=amount,account_run=self.receipt['account_run']))
            return count

    def finish(self,reservation,outcome):
        if outcome not in ('response','network_error','model_error'):
            raise ValueError('Unknown outcome')
        with self.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self._append(dict(event=outcome,id=reservation,at=time.time()))

    def status(self,task='',run=''):
        if task and task!=self.receipt['task']:
            raise BudgetStopped('reservation_task_mismatch')
        rows=self._read()
        reservations=[x for x in rows if x['event']=='reserved']
        if run and any(x['run']!=run for x in reservations):
            raise BudgetStopped('episode_already_attempted')
        return dict(account_run=self.receipt['account_run'],
                    limits=dict(episode_reserved_usd=self.receipt['cap_usd'],
                                cumulative_authorized_usd=self.receipt['cumulative_cap_usd']),
                    reserved=dict(run_calls=len(reservations),
                                  call_estimate_usd=.0015*len(reservations),
                                  account_reserved_usd=self.receipt['cumulative_reserved_usd']),
                    scope='This log spends against the existing account-ledger episode reservation; estimates are not settled charges.')
