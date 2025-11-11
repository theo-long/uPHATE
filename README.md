# Notes

- Looks like phate kernel matrix isn't actually symmetric? We divide out the *rows* by local bandwidth, but don't do this symmetrically with columns
    - for now we follow the existing library implementation and don't make it symmetric
- Need to handle landmarking