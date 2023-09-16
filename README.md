# HMQA_tryout
The repository focuses on the impletation of the improved ST-GCN for Human Motion Quaility Assessment. Within this repository, you will find code for solving this problem through replacing the LSTM module with Transformer module for avoiding long-term memory loss, and adding sparsity constraints for multii-task(classification task) to enlarge the dataset for single motion.

## Files: (Folders)
- 1-baseline_GCN: the baseline constructed by myself for future comparision according to the outline provided by this paper: "Graph_Convolutional_Networks_for_Assessment_of_Physical_Rehabilitation_Exercises"
- 2-improved_GCN-AGCN: replaced the LSTM module with Transformer module and added sparsity constraints for multii-task(classification task) to enlarge the dataset for single motion.
- 3-(paper_1_reference-code)STGCN-rehab-main: Source code for the paper: "Graph_Convolutional_Networks_for_Assessment_of_Physical_Rehabilitation_Exercises"
- 4-(paper_2_reference-code)ST-TR-master: Source code for the paper: "Skeleton-based action recognition via spatial and temporal transformer networks"

## Notes:
- Currently, all the content(esp. comments) in this project is written in Chinese and this project is still in the process of organization. **I cannot guarantee its immediate usability.**
- Some troubling but interesting problems when researching on this topic are shown in "Conclusion Report.docx"(written in Chinese), which I'll leave for future exploration.
