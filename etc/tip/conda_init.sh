# >>> conda initialize >>>
do_conda=1
if [ ${do_conda} -eq 1 ];then
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/dding/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/dding/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/dding/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/dding/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
fi
# <<< conda initialize <<<

