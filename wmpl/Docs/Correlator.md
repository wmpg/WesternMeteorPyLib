# Correlating RMS data into trajectories

**`wmpl.Trajectory.CorrelateRMS`** scans a directory tree of RMS station data, finds groups of
observations that saw the same meteor, solves each group into a trajectory and an orbit, and
remembers what it has already done so it can be left running. It is the bulk counterpart to
[solving a single event](Trajectory.md): you point it at an archive rather than at one meteor.

The work splits into three stages, which can run in one process, as separate processes on one
server, or spread across several servers. This page covers all three arrangements.

## Table of Contents

- [Quick start](#quick-start)
- [Operation modes](#operation-modes)
- [What it reads](#what-it-reads)
- [What it writes](#what-it-writes)
- [Command line reference](#command-line-reference)
- [Continuous processing on one server](#continuous-processing-on-one-server)
- [Distributed processing](#distributed-processing)
- [Setting up a child node](#setting-up-a-child-node)
- [The remote configuration file](#the-remote-configuration-file)
- [Running and managing the nodes](#running-and-managing-the-nodes)
- [The databases](#the-databases)
- [Inspecting the databases with CorrelateDB](#inspecting-the-databases-with-correlatedb)
- [Worked examples](#worked-examples)
- [Troubleshooting](#troubleshooting)

---

## Quick start

Run every stage in one pass over a data directory:

```
python -m wmpl.Trajectory.CorrelateRMS /path/to/rms_data
```

That is the form to use for a one-off analysis. To keep a live archive up to date, run it
continuously over the last few days of data:

```
python -m wmpl.Trajectory.CorrelateRMS /path/to/rms_data -a 5 --autofreq 60
```

For a busy archive, split the stages into three processes that run at their own cadences. Each one
needs its own log file, which is what `--addlogsuffix` is for:

```
python -m wmpl.Trajectory.CorrelateRMS /path/to/rms_data --mcmode 4 -a 3 --autofreq 15 --addlogsuffix
python -m wmpl.Trajectory.CorrelateRMS /path/to/rms_data --mcmode 1 -a 3 --autofreq 30 --addlogsuffix
python -m wmpl.Trajectory.CorrelateRMS /path/to/rms_data --mcmode 2 -a 3 --autofreq 30 --addlogsuffix
```

## Operation modes

`--mcmode` is a **bitmask**, not a list of alternatives. Three stages can be switched on
independently and combined by adding their values:

| Mode | Constant | What it does | Reads | Writes |
| :--- | :--- | :--- | :--- | :--- |
| `4` | `MCMODE_CANDS` | Finds groups of observations that saw the same meteor | The station folders | `candidates/` |
| `1` | `MCMODE_PHASE1` | Solves each candidate without Monte Carlo | `candidates/` | `phase1/` and `trajectories/` |
| `2` | `MCMODE_PHASE2` | Re-solves with the full Monte Carlo uncertainty run | `phase1/` | `trajectories/` |

Adding them gives the combined modes:

| Value | Constant | Runs | Use it when |
| :--- | :--- | :--- | :--- |
| `5` | `MCMODE_SIMPLE` | Candidates, then simple solutions | You want fast solutions and no uncertainties |
| `3` | `MCMODE_BOTH` | Simple solutions, then Monte Carlo | Candidates are produced elsewhere |
| `7` | `MCMODE_ALL` | Everything | The default full pipeline |
| `0` | — | Everything, remapped to `7` | The flag is omitted, or a one-off analysis |

**In the combined modes the intermediate files are not written.** A run with `--mcmode 5` keeps the
candidates in memory and goes straight on to solving them, so `candidates/` stays empty. Only the
single-stage modes hand work to the next stage through the file system, which is what makes the
stages separable across processes and servers.

**An unrecognised value is treated as a full run.** `getMcModeStr` only knows 0, 1, 2, 3, 4, 5 and
7, and anything else falls through to the full pipeline with a log suffix of `MIXED`.

## What it reads

The data directory holds one folder per station, each containing one folder per night, and each of
those an RMS `FTPdetectinfo` file together with the recalibrated plate solutions:

```
rms_data/
├── UK002F/
│   └── UK002F_20260712_210544_527636/
│       ├── FTPdetectinfo_UK002F_20260712_210544_527636.txt
│       └── platepars_all_recalibrated.json
├── UK008F/
│   └── UK008F_20260712_211431_942892/
│       ├── FTPdetectinfo_UK008F_20260712_211431_942892.txt
│       └── platepars_all_recalibrated.json
└── wmpl_remote.cfg          (only for distributed processing)
```

A night folder is skipped unless both files are present. The correlator walks the tree on every
pass, so stations and nights can be added while it is running.

`wmpl_remote.cfg` is read from `--dbdir` (the data directory unless you move it) at the start of
**every** pass, so changing node capacities does not need a restart.

## What it writes

Under `--outdir`, which defaults to the data directory:

```
trajectories/
└── 2026/
    └── 202607/
        └── 20260712/
            └── 20260712_221506.294_UK/
                ├── 20260712_221506_trajectory.pickle
                ├── 20260712_221506_report.txt
                └── ... plots
candidates/
└── processed/
phase1/
└── processed/
```

| Path | Contents |
| :--- | :--- |
| `trajectories/YYYY/YYYYMM/YYYYMMDD/` | One folder per solved trajectory, named from its reference time to the millisecond and the sorted two-letter country codes of its stations, e.g. `20260712_221506.294_UK` or `20260712_221506.294_BE_FR` |
| `candidates/` | Candidate observation groups waiting for a simple solution |
| `phase1/` | Simple solutions waiting for the Monte Carlo stage |
| `*/processed/` | Inputs already consumed, kept for a while so a crash can be recovered |

Consumed inputs are not kept forever. Processed candidates are removed after 21 days and processed
phase 1 solutions after twice that, both measured against the lookback window when `-a` or `-r` is
given.

The log file is `correlate_rms_<YYYYmmdd_HHMMSS>.log` in `--logdir`, rotated at midnight with seven
days kept. With `--addlogsuffix` the stage name is appended, so concurrent instances do not share a
file:

| `--mcmode` | Log file suffix |
| :--- | :--- |
| `4` | `_cands` |
| `1` | `_simple` |
| `2` | `_mcphase` |
| `5` | `_candsimple` |
| `3` | `_simplemc` |
| `0` or `7` | `_full` |

## Command line reference

```
python -m wmpl.Trajectory.CorrelateRMS --help
```

| Flag | Description |
| :--- | :--- |
| `dir_path` | **Required.** Root data directory. Helper files are written here unless redirected. |
| `-t`, `--maxtoffset` | Maximum time offset between stations, in seconds. Default: `10`. |
| `-s`, `--maxstationdist` | Maximum distance between paired stations, in km. Default: `600`. |
| `-m`, `--minerr` | Error in arcsec below which a station is never rejected. Default: `30`. |
| `-M`, `--maxerr` | Error in arcsec above which a station is rejected. Default: `180`. |
| `-v`, `--maxveldiff` | Maximum velocity difference between two stations, in percent. Default: `25`. |
| `-p`, `--velpart` | Fraction from the start of the meteor used for the initial velocity fit. Default: `0.4`; raise to `0.5` for noisy data. |
| `-x`, `--maxstations` | Use only the best N stations in a solution. Default: `15`. |
| `-d`, `--disablemc` | Disable Monte Carlo. |
| `-u`, `--uncerttime` | Cull solutions with a worse time fit than the line-of-sight solution. Considerably slower. |
| `-l`, `--saveplots` | Save plots to disk. |
| `-o`, `--enableOSM` | Use OpenStreetMap ground plots. Needs an internet connection. |
| `-r`, `--timerange` | Only process a time range, as `"(YYYYMMDD-HHMMSS,YYYYMMDD-HHMMSS)"`. |
| `-a`, `--auto` | Run continuously over the last `PREV_DAYS` days. Default when given without a value: `5`. |
| `--autofreq` | Minutes between passes in auto mode. Default: `360`. |
| `--cpucores` | Cores to use. `-1` means all but one. Default: `-1`. |
| `--mcmode` | Stage bitmask. See [Operation modes](#operation-modes). Default: `0`, meaning everything. |
| `--maxtrajs` | Most candidates or solutions to load per pass in a single-stage run. Default: unlimited. |
| `--dbdir` | Where the databases and `wmpl_remote.cfg` live. Default: the data directory. |
| `--outdir` | Where trajectories and intermediate files go. Default: the data directory. |
| `--logdir` | Where logs go. Default: the output directory. |
| `--archivemonths` | Months of history to keep before archiving. Default: `0`, which purges instead. |
| `--addlogsuffix` | Append the stage name to the log file name. |
| `--verbose` | Verbose logging. |

**`--archivemonths` defaults to purging, not archiving.** With the default `0`, records older than
the retention window are deleted rather than moved into a monthly archive database. Retention never
drops below 21 days, and when `-r` is given the cutoff is the earlier of the two.

**`--remotehost` has been removed.** Distributed processing is configured entirely through
`wmpl_remote.cfg`. Command lines that still pass `--remotehost` now fail to parse; delete the flag
and write a configuration file instead.

## Continuous processing on one server

One process running everything is simplest, but the three stages have very different throughputs, so
a busy archive is better served by three processes at three cadences. Measured on a 16-core server:

| Stage | Throughput | Suggested `--autofreq` | Suggested `--maxtrajs` |
| :--- | :--- | :--- | :--- |
| `--mcmode 4` candidates | ~6000 candidates/hour | 15 | not needed |
| `--mcmode 1` simple | ~1000 candidates/hour | 30 | 200 |
| `--mcmode 2` Monte Carlo | ~200 solutions/hour | 30 to 60 | 200 |

The Monte Carlo stage dominates the cost, which is why it is the one worth moving onto other
machines.

## Distributed processing

In distributed mode a **parent** node owns everything durable and one or more **child** nodes own
nothing. The parent holds the raw data, the output tree and the databases; children receive work,
solve it, and hand the results back.

The transport is **SFTP over SSH**, and the parent never connects outwards. Each child has a Unix
account on the parent, and the parent hands out work simply by writing files into that account's
home directory on its own file system. The child logs in over SFTP, pulls the work, and pushes the
results back into the same place:

```
parent  --writes-->  /home/node1/files/candidates/   --child pulls-->  child solves
child   --pushes-->  /home/node1/files/{phase1,trajectories}/ and its own *.db  --parent merges-->  parent
```

A child runs exactly one stage, either `--mcmode 1` or `--mcmode 2`, chosen per node in the
configuration file. `paramiko` provides the SFTP client and is already a requirement of the library.

## Setting up a child node

This is a one-time setup, and most of it happens on the parent. The commands are for Linux; a child
can also run under WSL2 on Windows.

### Create a key on the child

On the child machine, create a key pair and copy the public half to the parent:

```
ssh-keygen -t ed25519 -f ~/.ssh/wmpl -C "wmpl node1"
scp ~/.ssh/wmpl.pub parent:/tmp/node1.pub
```

### Create the account and folders on the parent

Give every child its own account, so nodes cannot see each other's work:

```
sudo groupadd wmpl
sudo useradd -m -g wmpl -s /bin/bash node1
sudo mkdir -p ~node1/files/candidates/processed ~node1/files/phase1/processed ~node1/files/trajectories
sudo chown -R node1:wmpl ~node1/files
sudo chmod -R 775 ~node1/files
```

The resulting layout is what the parent and child agree on:

```
/home/node1/
└── files/
    ├── candidates/        work assigned to this node
    │   └── processed/     candidates it has consumed
    ├── phase1/            simple solutions, in either direction
    │   └── processed/     phase 1 work it has consumed
    ├── trajectories/      finished trajectories waiting to be collected
    └── stop               present means "assign no more work"
```

### Grant the parent account access

The account that runs the correlator must be able to read and write inside the child's home.
Group membership alone is not enough, because the home directory itself is owned by the child. Use a
POSIX ACL, substituting the account the correlator runs as:

```
sudo setfacl -R -m u:ubuntu:rwx ~node1
sudo setfacl -R -d -m u:ubuntu:rwx ~node1
```

The solver sets its own permissions on the files it creates. Do not change them.

### Enable the login and hold the node back

Install the child's public key, then create the `stop` file so the parent does not start assigning
work before the child is ready:

```
sudo -u node1 mkdir -p ~node1/.ssh
sudo -u node1 sh -c 'cat /tmp/node1.pub >> ~/.ssh/authorized_keys'
sudo -u node1 chmod 700 ~node1/.ssh && sudo -u node1 chmod 600 ~node1/.ssh/authorized_keys
sudo -u node1 touch ~node1/files/stop
```

Check the login from the child before going further:

```
ssh -i ~/.ssh/wmpl node1@parent
```

## The remote configuration file

Copy `wmpl_remote.cfg.sample` from the repository root to `wmpl_remote.cfg` in the directory given
by `--dbdir`. **Every node needs its own copy**, and it is re-read on every pass, so capacities can
be changed while the system runs.

| Section | Key | Meaning |
| :--- | :--- | :--- |
| `[mode]` | `mode` | `parent` (or `master`) or `child`. Decides which of the other sections is used. |
| `[children]` | `<name>` | One line per child node, read by the parent. See below. |
| `[sftp]` | `host`, `user`, `key` | How a child reaches the parent. `port` is optional and defaults to 22. |

Each child line is three comma-separated fields:

| Field | Meaning |
| :--- | :--- |
| `dirpath` | The node's home directory on the parent, whose `files/` folder the parent writes into |
| `capacity` | How many items may be queued for it. `0` disables the node, `-1` means unlimited |
| `mcmode` | `1` for the simple stage or `2` for the Monte Carlo stage |

On the parent:

```
[mode]
mode = parent

[children]
node1 = /home/node1,500,1
node2 = /home/node2,400,1
node3 = /home/node3,200,2
```

On a child:

```
[mode]
mode = child

[sftp]
host = parent.example.com
user = node1
key = ~/.ssh/wmpl
```

A node whose `files/` folder does not exist, or is not readable by the correlator's account, is
treated as having capacity 0 and is logged as skipped. As a starting point on a 4-core desktop, 500
suits the simple stage and 200 the Monte Carlo stage, which takes roughly three hours for 200
solutions.

## Running and managing the nodes

### Starting a child

A child runs the same program, in a single stage, with its own log:

```
python -m wmpl.Trajectory.CorrelateRMS /path/to/work --mcmode 1 -a 3 --autofreq 15 --addlogsuffix
```

Starting up clears the `stop` file, so the parent begins assigning work on its next pass.

**Do not pass `--maxtrajs` on a child.** The parent fills the node up to its configured capacity; if
the child then only loads a smaller number per pass, the difference accumulates as a backlog that
never clears.

### The stop file

A file called `stop` in the node's `files/` folder means "assign no more work". The child creates it
when interrupted with Ctrl-C and on a clean exit, and you can create it by hand at any time. When
the parent sees it, it pulls back everything still waiting in that node's folders.

### How work is assigned

The parent tops a node up only when its target folder is empty, so work is handed out in batches
rather than trickling. At most half of the pending pool moves at once, an unlimited capacity is
still capped at 1000 items, and anything already marked as being processed is skipped. If no node
can take the work, it stays on the parent and is solved there.

### Reclaiming stale work

Anything left in a node's `candidates/` or `phase1/` folder with a modification time more than six
hours old is assumed abandoned and is moved back to the parent.

### After a child crashes

Work the child had already downloaded sits in its `processed/` folders. Move the suspect batch back
so it is picked up again:

```
ls -1 ~node1/files/candidates/processed | head -100 | while read f
do
    mv ~node1/files/candidates/processed/$f ~node1/files/candidates/
done
```

To find out which node had a given trajectory, search the parent's log:

```
egrep "node1|node2" correlate_rms_*_cands.log | grep saving
```

### Distributing the Monte Carlo stage

Add a node with mode `2` in `[children]` and start the child with `--mcmode 2`. Since the Monte
Carlo stage is by far the most expensive, this is usually where extra machines pay off first.

## The databases

Three SQLite databases in `--dbdir` replace the single `processed_trajectories.json` used
previously. They are split deliberately: SQLite allows only one writer at a time, and separating the
three kinds of record keeps the stages from blocking each other.

| Database | Tables | Written mainly by |
| :--- | :--- | :--- |
| `observations.db` | `paired_obs` | The simple stage |
| `trajectories.db` | `trajectories`, `failed_trajectories` | The simple and Monte Carlo stages |
| `candidates.db` | `candidates` | The candidate stage |

All three use write-ahead logging. Every row carries a status:

| `status` | Meaning |
| :--- | :--- |
| `1` | Live |
| `0` | Deleted. The row is kept until it is purged or archived |
| `2` | Being processed right now |

Rows stuck at `2` after a crash are reset to `1` at the next startup, so the work is retried.

### Migrating from the JSON database

The conversion is automatic and happens once. On startup, if **neither** `observations.db` **nor**
`trajectories.db` exists, the legacy `processed_trajectories.json` is read and its recent records
are copied across. Only the lookback window is copied, which is five days under a plain `-a`. The
JSON file is then never read again and is left in place as a historical record.

### Historic reruns

Because only the window was copied, a run over data from before the cutover finds no paired
observations and no failed trajectories, and reanalyses the raw data from scratch. To avoid that,
backfill the databases for the period you are about to process:

```
python -m wmpl.Trajectory.CorrelateDB --dir_path rms_data --action copy --timerange "(20260101-000000,20260401-000000)"
```

This takes a few minutes per week of data.

### Archiving and retention

With `--archivemonths N`, records older than N months are moved into monthly archive databases named
`YYYYMM_observations.db` and so on. With the default `0` they are deleted instead. Retention is
never shorter than 21 days. Candidates are never archived, since they are already capped at 21 days.

### Write-ahead logs

After a crash you may find `*.db-wal` and `*.db-shm` files next to a database. **Do not delete
them.** SQLite completes or rolls back the outstanding transaction the next time the database is
opened; at worst a few observations are processed twice.

### Duplicates and mergeable trajectories

Two solutions that share **all** their observations are duplicates: the one with the fewest ignored
observations is kept and the others are removed from the database and from disk. Two solutions that
share **some** observations are mergeable, which typically happens when a camera uploads late and a
second, better solution appears for a meteor already solved. Again the better solution is kept.

## Inspecting the databases with CorrelateDB

`wmpl.Trajectory.CorrelateDB` is both the database layer and a small administrative command line.

| Flag | Description |
| :--- | :--- |
| `--dir_path` | **Required.** Directory holding the databases. |
| `--database` | `observations` or `trajectories`. The candidate database is not reachable from here. |
| `--action` | `status`, `copy` or `execute`. |
| `--stmt` | The SQL to run with `--action execute`. |
| `-r`, `--timerange` | Time range for `--action copy`. |
| `--logdir` | Where to write the log. Defaults to a `logs` folder beside the databases. |

| Action | What it does |
| :--- | :--- |
| `status` | Prints how many rows each table holds. Needs `--database` |
| `copy` | Backfills both databases from the legacy JSON over `--timerange`. Ignores `--database` |
| `execute` | Runs the statement in `--stmt`. Needs `--database` |

```
python -m wmpl.Trajectory.CorrelateDB --dir_path rms_data --database trajectories --action status
python -m wmpl.Trajectory.CorrelateDB --dir_path rms_data --database trajectories --action execute --stmt "select count(*) from trajectories where status=2"
```

`--action execute` runs raw SQL against a live database, so treat it as an administrator's tool.

## Worked examples

**Reprocess one night.** No auto mode, one pass, everything in one process:

```
python -m wmpl.Trajectory.CorrelateRMS rms_data -r "(20260712-120000,20260713-120000)"
```

**A candidate finder on the parent**, looking three days back, every quarter of an hour:

```
python -m wmpl.Trajectory.CorrelateRMS rms_data --mcmode 4 -a 3 --autofreq 15 --addlogsuffix
```

**A simple-solution stage**, capped so a pass does not run unboundedly long:

```
python -m wmpl.Trajectory.CorrelateRMS rms_data --mcmode 1 -a 3 --autofreq 30 --maxtrajs 500 --addlogsuffix
```

**A Monte Carlo stage** on a dedicated machine, with plots and an archive cutover at six months:

```
python -m wmpl.Trajectory.CorrelateRMS rms_data --mcmode 2 -a 3 --autofreq 30 --maxtrajs 200 \
    --archivemonths 6 --saveplots --addlogsuffix
```

## Troubleshooting

**`unrecognized arguments: --remotehost`.** The flag was removed. Configure the nodes in
`wmpl_remote.cfg` in the `--dbdir` directory instead.

**`Permission denied (publickey)` when a child connects.** The child's public key is not in
`~/.ssh/authorized_keys` of its account on the parent, or that file's permissions are too open. It
must be mode 600 inside a mode 700 `.ssh` directory owned by the child account.

**The parent logs that a node is skipped because it has no `files` folder.** The correlator's
account cannot see into the child's home directory. Group membership is not sufficient, because the
home directory belongs to the child account; apply the `setfacl` ACL shown above.

**A child downloads work but its backlog keeps growing.** `--maxtrajs` is set on the child below the
capacity configured for it on the parent. Remove `--maxtrajs` from the child's command line.

**A node is never given any work.** Check, in this order: the `stop` file left over from setup, a
capacity of `0`, a node `mcmode` that does not match the stage producing work, and a target folder
that is not yet empty, since a node is only topped up once it has drained.

**`candidates/` stays empty although the correlator is running.** The combined modes keep
intermediate results in memory. Only `--mcmode 4`, `1` and `2` individually write the handover files.

**A `--mcmode` value behaves like a full run.** Only 0, 1, 2, 3, 4, 5 and 7 are recognised; anything
else falls through to the full pipeline and logs its stage as `MIXED`.

**A historic rerun resolves everything from scratch.** The migration only copied the lookback
window, so earlier observations look unpaired. Backfill with `CorrelateDB --action copy` over the
period first.

**The JSON database was not migrated.** Migration only runs when neither `observations.db` nor
`trajectories.db` exists. Once either is present the JSON file is ignored; use
`CorrelateDB --action copy` to bring records across afterwards.

**`*.db-wal` files are left after a crash.** Leave them alone and restart. Deleting them discards a
transaction SQLite is able to finish on its own.

**Two instances write to the same log file.** Pass `--addlogsuffix` to every instance so each stage
gets its own file.

---

Related: [solving a single trajectory](Trajectory.md), and
[integrating the resulting orbits](REBOUND.md).
