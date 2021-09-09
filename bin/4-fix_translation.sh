#!/bin/bash
for f in out/*.translated.en-is*; do
    cat $f |
        sed 's|<L> |<L>|g' | \
        sed 's| </L>|</L>|g' | \
        sed 's|<O> |<O>|g' | \
        sed 's| </O>|</O>|g' | \
        sed 's|<P> |<P>|g' | \
        sed 's| </P>|</P>|g' | \
        sed 's|<M> |<M>|g' | \
        sed 's| </M>|</M>|g' > $f.fixed
done
