"""Split manifolds, one-step gaps, event boundaries and immutable failure tests."""
from contact_damage import ContactPoint, ContactDamage

def verify(hz):
    def fresh(): return ContactDamage(hz,[1,2,3],[(1,2)],[])
    def p(a,b,n,v=0.,sep=0.): return ContactPoint(a,b,sep,n/hz,v)
    g=fresh()
    assert g.observe(1,[p(1,3,80),p(3,1,80)])[-1]=="force_n"
    assert g.observe(2,[])[0]==1
    g=fresh()
    assert g.observe(1,[p(1,2,80),p(2,1,80)]) is None
    assert g.observe(2,[p(2,1,0,sep=-.00201)])[-1]=="penetration_m"
    g=fresh()
    assert g.observe(1,[p(1,3,100,-.03*hz)]) is None  # 3 joules
    assert g.observe(2,[]) is None
    assert g.observe(3,[p(3,1,100,-.03*hz)])[-1]=="event_work_j"
    g=fresh()
    assert g.observe(1,[p(1,3,100,-.03*hz)]) is None
    assert g.observe(2,[]) is None and g.observe(3,[]) is None
    assert g.observe(4,[p(3,1,100,-.03*hz)]) is None
    assert g.observe(5,[p(1,3,0,sep=float("nan"))])[-1]=="non-finite"
    return hz

if __name__=="__main__":
    print("contact damage controls passed at",verify(240),verify(480),"Hz")
