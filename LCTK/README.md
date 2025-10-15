# How to use

## apply patches

```bash
cd [YOUR LCTK DIR]
patch -p1 < [THIS DIR]/patches/*
```

## reverse patches

```bash
cd [YOUR LCTK DIR]
PATCHDIR=[THIS DIR]/patches; for d in $(ls $PATCHDIR | tac); do patch -R -p1 < "$PATCHDIR/$d"; done
```

