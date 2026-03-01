# LearnAgent Curriculum Report

## Summary

```json
{
  "config": {
    "episodes_per_task": 3,
    "strategies": [
      "baseline",
      "beam",
      "critic"
    ]
  },
  "tasks": [
    {
      "task_id": "users_events_v1",
      "runs": [
        {
          "strategy": "baseline",
          "i": 0,
          "ok": false
        },
        {
          "strategy": "baseline",
          "i": 1,
          "ok": false
        },
        {
          "strategy": "baseline",
          "i": 2,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 0,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 1,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 2,
          "ok": false
        },
        {
          "strategy": "critic",
          "i": 0,
          "ok": false
        },
        {
          "strategy": "critic",
          "i": 1,
          "ok": true
        },
        {
          "strategy": "critic",
          "i": 2,
          "ok": true
        }
      ]
    },
    {
      "task_id": "mean_csv_v1",
      "runs": [
        {
          "strategy": "baseline",
          "i": 0,
          "ok": false
        },
        {
          "strategy": "baseline",
          "i": 1,
          "ok": false
        },
        {
          "strategy": "baseline",
          "i": 2,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 0,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 1,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 2,
          "ok": false
        },
        {
          "strategy": "critic",
          "i": 0,
          "ok": true
        },
        {
          "strategy": "critic",
          "i": 1,
          "ok": true
        },
        {
          "strategy": "critic",
          "i": 2,
          "ok": true
        }
      ]
    },
    {
      "task_id": "report_md_v1",
      "runs": [
        {
          "strategy": "baseline",
          "i": 0,
          "ok": false
        },
        {
          "strategy": "baseline",
          "i": 1,
          "ok": false
        },
        {
          "strategy": "baseline",
          "i": 2,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 0,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 1,
          "ok": false
        },
        {
          "strategy": "beam",
          "i": 2,
          "ok": false
        },
        {
          "strategy": "critic",
          "i": 0,
          "ok": true
        },
        {
          "strategy": "critic",
          "i": 1,
          "ok": true
        },
        {
          "strategy": "critic",
          "i": 2,
          "ok": true
        }
      ]
    }
  ],
  "by_strategy": {
    "baseline": {
      "success_rate": 0.0,
      "n": 9
    },
    "beam": {
      "success_rate": 0.0,
      "n": 9
    },
    "critic": {
      "success_rate": 0.8888888888888888,
      "n": 9
    }
  },
  "mined_rules_added": [
    "Ensure required CSV files exist before running data processing scripts.",
    "Create missing CSV files with necessary headers and sample data to avoid runtime errors.",
    "Use set operations in Python to find unique elements across multiple CSV files.",
    "Skip header rows when reading CSV files to process only the data rows.",
    "Verify file paths and existence of files used in scripts to prevent FileNotFoundError."
  ]
}
```
