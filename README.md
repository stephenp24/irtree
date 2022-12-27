# sconfig

Welcome to `sconfig`, a library for easily resolving complex data inheritance written in Python.

## Introduction

Inheritance is a powerful feature that allows one set of data to inherit attributes from another set data. However, when dealing with complex inheritance hierarchies, it can be difficult to correctly resolve the inherited data. This is where `sconfig` comes in.

## Features

- Provides a simple and intuitive API for resolving inherited data
- Handles multiple inheritance (both `direct` and `non-direct`) scenarios
- Allows customization of the `resolving`
- Allows customization of the `data item` 
- Extensively tested to ensure correctness and reliability

## Installation

### Requirements

- python `poetry`

### From source

- `git clone` this project
  
  > git clone https://github.com/stephenp24/sconfig.git

- `cd` to the sconfig dir

  > cd sconfig

- `poetry install` and let it find all the requirements

  > `poetry install --without test,docs`

### Add as dependencies to your other poetry project

- Local repo, use poetry [add](https://python-poetry.org/docs/cli/#add)

  > poetry add ../sconfig

- Git dependencies, use poetry [add](https://python-poetry.org/docs/cli/#add)

  > poetry add git+ssh://git@github.com:stephenp24/sconfig.git

## Basic usage

`sconfig` came with a `ConfigNode` class that should handle most of data inheritance cases, to use `ConfigNode` simply construct one.
