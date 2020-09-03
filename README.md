This repo is a proof of concept, and contains code to train quantum neural nets on training data, by
analytically computing their loss gradient and running gradient descent.

The inspiration of this approach was [this paper]{https://arxiv.org/abs/1810.13295}, which
restricted each gate to be in a specific 1-parameter family of quantum gates, which I generalized to
allow each gate to be any quantum gate at all. See writeup.pdf for a high-level explanation of the
ideas behind this, and an unfortunate story about why this technique isn't actually practical.

`optimize_qca.py` contains the code for optimizing a QCA, and `mod_qubit.py` and `move_qubit.py`
contain simple examples of using it to train QCAs for solving simple problems; even these problems
will exhibit the above issues if pushed to sufficiently many qubits.
