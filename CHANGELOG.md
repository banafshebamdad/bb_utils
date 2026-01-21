# Changelog

All notable changes to bb_utils will be documented in this file.

## [0.1.0] - 2026-01-21

### Added
- **InCrowd-VI Label Generator**: Complete implementation for generating SuperPoint training labels
  - Processes semi-dense SLAM observations from InCrowd-VI dataset
  - Outputs labels in SuperPoint-compatible format (.npz files)
  - Configurable confidence computation (inverse_normalized, exponential)
  - Memory-efficient chunked CSV processing
  - Automatic report generation with statistics
  - Comprehensive documentation and examples
  
- **NPZ Inspector Updates**: Enhanced per-column statistics
  - Added per-column min/max/mean for 2D arrays
  - Improved output formatting
  - Better handling of multi-dimensional data
  
- **Package Infrastructure**:
  - Console script entry points (`generate-incrowdvi-labels`, `inspect-npz`)
  - YAML configuration support
  - Example scripts and usage documentation
  - Quick start guide
  - Implementation summary

- **Dependencies**:
  - pandas >= 1.5.0
  - PyYAML >= 6.0
  - tqdm >= 4.65.0

### Changed
- Updated README.md with comprehensive documentation
- Enhanced package structure with configs/, examples/, and docs/ directories
- Improved pyproject.toml with all dependencies and scripts

## [0.0.1] - Initial Version

### Added
- Basic NPZ file inspector
- Package structure
- Initial README
