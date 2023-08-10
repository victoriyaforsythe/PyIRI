Contributing
============

Bug reports, feature suggestions, and other contributions are greatly
appreciated!  PyIRI is a community-driven project and welcomes both feedback and
contributions.

Short version
-------------

* Submit bug reports and feature requests at
  [GitHub](https://github.com/victoriyaforsythe/PyIRI/issues)

* Make pull requests to the ``develop`` branch

Bug reports
-----------

When [reporting a bug](https://github.com/victoriyaforsythe/PyIRI/issues) please
include:

* Your operating system name and version

* Any details about your local setup that might be helpful in troubleshooting

* Detailed steps to reproduce the bug

Feature requests and feedback
-----------------------------

The best way to send feedback is to file an issue at
[GitHub](https://github.com/victoriyaforsythe/PyIRI/issues).

If you are proposing a feature:

* Explain in detail how it would work.

* Keep the scope as narrow as possible, to make it easier to implement.

* Remember that this is a volunteer-driven project, and that code contributions
  are welcome :)

Development
-----------

To set up `PyIRI` for local development:

1. [Fork PyIRI on GitHub](https://github.com/victoriyaforsythe/PyIRI/fork).

2. Clone your fork locally:

  ```
    git clone git@github.com:your_name_here/PyIRI.git
  ```

3. Create a branch for local development:

  ```
    git checkout -b name-of-your-bugfix-or-feature
  ```

   Now you can make your changes locally.

4. When you're done making changes, run all the checks to ensure that nothing
  is broken on your local system:

  ```
  pytest PyIRI
  ```

5. You should also check for flake8 style compliance:

  ```
  flake8 . --count --select=D,E,F,H,W --show-source --statistics
  ```

  Note that PyIRI uses the `flake-docstrings` and `hacking` packages to ensure
  standards in docstring formatting.


6. Update/add documentation (in ``docs``), if relevant

7. Add your name to the .zenodo.json file as an author

8. Commit your changes:
  ```
  git add .
  git commit -m "AAA: Brief description of your changes"
  ```
  Where AAA is a standard shorthand for the type of change (eg, BUG or DOC).
  `PyIRI` follows the [numpy development workflow](https://numpy.org/doc/stable/dev/development_workflow.html),
  see the discussion there for a full list of this shorthand notation.  

9. Once you are happy with the local changes, push to GitHub:
  ```
  git push origin name-of-your-bugfix-or-feature
  ```
  Note that each push will trigger the Continuous Integration workflow.

10. Submit a pull request through the GitHub website. Pull requests should be
   made to the ``develop`` branch.  Note that automated tests will be run on
   github actions, but these must be initialized by a member of the PyIRI team.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code, just
make a pull request. Pull requests should be made to the ``develop`` branch.

For merging, you should:

1. Include an example for use
2. Add a note to ``CHANGELOG.md`` about the changes
3. Update the author list in ``zenodo.json`` if applicable
4. Ensure that all checks passed (current checks include Github Actions and
   Coveralls)

If you don't have all the necessary Python versions available locally or have
trouble building all the testing environments, you can rely on GitHub Actions
to run the tests for each change you add in the pull request. Because testing
here will delay tests by other developers, please ensure that the code passes
all tests on your local system first.


Project Style Guidelines
------------------------

In general, PyIRI follows PEP8 and numpydoc guidelines.  Pytest runs the unit
and integration tests, flake8 checks for style, and sphinx-build performs
documentation tests.  However, there are certain additional style elements that
have been adopted to ensure the project maintains a consistent coding style.
These include:

* Line breaks should occur before a binary operator (ignoring flake8 W503)
* Combine long strings using `join`
* Preferably break long lines on open parentheses rather than using `\`
* Use no more than 80 characters per line
* Several dependent packages have common nicknames, including:
  * `import datetime as dt`
  * `import numpy as np`
* All classes should have `__repr__` and `__str__` functions
* Try to avoid creating a try/except statement where except passes
* Block and inline comments should use proper English grammar and punctuation
  with the exception of single sentences in a block, which may then omit the
  final period
