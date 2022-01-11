---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# String formatting

```python
import physipy
from physipy import m
```

```python
q = 1.23456789*m
```

```python
print(str(q))
```

```python
q._compute_value()
```

## Standard formatting

```python
print(f"{q:.2f}")
print(f"{q:+.2f}")
print(f"{q:+9.2f}")
print(f"{q:*^15}")
print(f"{q: >-12.3f}")
```

## physipy formatting

```python
print(f"{q}")
print(f"{q:~}")  # ~ : no prefix before unit
```


```python

```

```python

```
