# HDG_postprocess
 A pakage to read and postprocess SOLEDGE-HDG solutions

## Modern additive API

The legacy API remains supported:

```python
from hdg_postprocess.formats import load_from_file

solution = load_from_file.load_HDG_solution_from_file(...)
```

An additive modern API is also available for new code:

```python
from hdg_postprocess.api import load_solution

solution = load_solution(...)
full_cons = solution.fields.conservative(view="full")
simple_phys = solution.fields.physical(view="simple")
profile = solution.sample.line(r_line, z_line, ["n", "te", "ti"])
power = solution.analysis.power_balance()
```

This modern layer is a thin facade over the legacy implementation, so it does not force migration and keeps existing demos working unchanged.
