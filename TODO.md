# General

### Todo

- [ ] Change to a better profiler, `line-profiler` doesnt seem to be a great choice
  - Aside from being too verbose at times, there's another another issue with 
    python coverage. When profiler is set, the coverage skips the function content
    producing invalid coverage results.
- [ ] Double check if dataclasses is a good choice.. Some drawbacks are:
  - enforce minimum python to `python-3.7`
  - some inheritance issue, e.g:
    - pre python-3.10, `kw_only` doesn't exactly works
    - dunder method doesn't ineherited properly
- [ ] Create performance tests and re-evaluate the performances, consider porting to cython 
  as needed.

# BaseDataItem

### Todo

- [ ] use `kw_only` when ported to 3.10

### In Progress ...

### Done

# BaseNode

### Todo

- [ ] use `kw_only` when ported to 3.10

### In Progress ...

### Done
