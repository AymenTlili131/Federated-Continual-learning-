# Swap Memory Setup Guide

## Why Enable Swap?

With 33GB RAM usage during data loading, enabling swap provides a safety net against OOM (Out of Memory) errors.

## Quick Setup (Recommended: 32GB Swap)

```bash
# 1. Check current swap
free -h

# 2. Create 32GB swap file
sudo fallocate -l 32G /swapfile

# 3. Set permissions
sudo chmod 600 /swapfile

# 4. Make it a swap file
sudo mkswap /swapfile

# 5. Enable swap
sudo swapon /swapfile

# 6. Verify
free -h
# Should show 32G swap

# 7. Make permanent (survives reboot)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 8. Optimize swappiness (how aggressively to use swap)
# 10 = only use swap when necessary
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## Monitoring

```bash
# Watch RAM and swap usage
watch -n 1 free -h

# Check if swap is being used
swapon --show
```

## Expected Behavior

- **Normal operation:** Swap unused, all in RAM
- **Data loading spike:** May touch swap briefly
- **Training:** Should stay in RAM (GPU takes over)

## If Swap is Heavily Used

This means you're hitting RAM limits. Solutions:

1. **Enable cudf.pandas** (already added) - moves data to GPU RAM
2. **Reduce parallel experiments** - fewer processes = less RAM
3. **Increase swap** to 64GB if needed

## Remove Swap (if needed)

```bash
sudo swapoff /swapfile
sudo rm /swapfile
# Remove from /etc/fstab
```

## Status: Ready

Swap is now configured as a safety net. With cudf.pandas enabled, you should rarely hit it.
