inline SplitPair KDTree::splitBoundingBox(const BoundingBox & box, SplitData type, float value) const
{
    SplitPair pair;
    for(int i=0;i<3;i++)
    {
		pair.left.min[i]=box.min[i];
		pair.left.max[i]=box.max[i];
		pair.right.min[i]=box.min[i];
		pair.right.max[i]=box.max[i];	
    }
    pair.left.max[type] = value;
    pair.right.min[type]  = value;	
    return pair;
}

inline SplitPair  KDTree::splitBoundingBox(const BoundingBox & box, KNode * node) const
{
    return splitBoundingBox(box,getType(node),getSplit(node));
}
