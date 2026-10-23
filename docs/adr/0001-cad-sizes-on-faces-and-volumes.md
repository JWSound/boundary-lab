# CAD import stores mesh sizes on faces and volumes only

The CAD import editor stores size overrides only on faces and on air-body volumes; a size chosen for a body or face group is written to each member, and those rows only show their members' shared size or "Mixed". Inherited sizes (face group, then body) made a row's shown size depend on rules the user could not see, and a volume size that also sized the body's faces made a face set to Global come out coarser. The headless pipeline still accepts body and face-group overrides, so older designs and scripts keep working; the editor converts them to face and volume sizes on load and never writes them again.

## Considered Options

- Keep inheritance (face, face group, body) with the inherited size shown greyed: rejected, a pick on a group did not reach faces with their own size.
- Volume interior following its faces (coarsest face): rejected, one coarse wall would coarsen the whole interior.
