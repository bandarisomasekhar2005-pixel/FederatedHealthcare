
"""Small additive homomorphic encryption demonstration.

This is a self-contained Paillier implementation used only for tiny integer
values. It demonstrates additive homomorphism: Dec(E(a)*E(b)) == a+b.
It is not used for neural-network training and is not production cryptography.
"""
import math, secrets

def _egcd(a,b):
    if b==0: return (a,1,0)
    g,x1,y1=_egcd(b,a%b); return (g,y1,x1-(a//b)*y1)

def _inv(a,m):
    g,x,_=_egcd(a,m)
    if g!=1: raise ValueError("inverse does not exist")
    return x%m

def _is_probable_prime(n, rounds=8):
    if n<2: return False
    for p in (2,3,5,7,11,13,17,19,23,29):
        if n%p==0: return n==p
    d=n-1;s=0
    while d%2==0: s+=1;d//=2
    for _ in range(rounds):
        a=secrets.randbelow(n-3)+2
        x=pow(a,d,n)
        if x in (1,n-1): continue
        for __ in range(s-1):
            x=pow(x,2,n)
            if x==n-1: break
        else: return False
    return True

def _prime(bits=256):
    while True:
        p=secrets.randbits(bits)|1|(1<<(bits-1))
        if _is_probable_prime(p): return p

def keygen(bits=512):
    p=_prime(bits//2); q=_prime(bits//2)
    while p==q: q=_prime(bits//2)
    n=p*q; nsq=n*n; g=n+1
    lam=math.lcm(p-1,q-1)
    mu=_inv((pow(g,lam,nsq)-1)//n,n)
    return (n,g), (lam,mu)

def encrypt(m, pub):
    n,g=pub; nsq=n*n
    if not (0 <= m < n): raise ValueError("demo supports non-negative integers below n")
    while True:
        r=secrets.randbelow(n)
        if math.gcd(r,n)==1: break
    return (pow(g,m,nsq)*pow(r,n,nsq))%nsq

def decrypt(c, pub, priv):
    n,g=pub; lam,mu=priv; nsq=n*n
    x=(pow(c,lam,nsq)-1)//n
    return (x*mu)%n

def demo(values=(17,25,8)):
    pub,priv=keygen(512)
    encrypted=[encrypt(int(v),pub) for v in values]
    aggregate=1
    for c in encrypted: aggregate=(aggregate*c)%(pub[0]**2)
    plain=decrypt(aggregate,pub,priv)
    return {"plain_values":list(values),"encrypted_count":len(encrypted),
            "decrypted_sum":int(plain),"expected_sum":int(sum(values)),
            "homomorphic_addition_verified":int(plain)==int(sum(values))}
