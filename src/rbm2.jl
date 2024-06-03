


module SciKit;

  using Statistics, CSV, DataFrames;

  export update;


  
  function initW(s1, s2, d1, d2 =[0])

    return randn(s1,s2) .* d1 .+ rand(d2, s1,s2);

    end;


 
  function sigmoid(z)

    return 1.0 / (1.0 + 2.4 ^ -z);

    end;


  function forward(x, w)

    return x * w;

    end;


  function step(x, w)

    o1 = sigmoid.(forward(x,w'));

    i2 = forward(o1, w);

    o2 = sigmoid.(forward(i2,w'));

    p1 = forward(x', o1)';

    n1 = forward(i2', o2)';

    return p1 .- n1;

    end;



  function ae_Step(data, w, w2)

    n = size(data,1);

    l1 = sigmoid.(forward(data, w'));  #

    l1_deriv = l1 .* (1.0 .- l1);

    out1 = forward(l1, w2);            #

    err1 = data .- out1;   ### 

    err2 = forward(err1, w2');         #

    err2b = err2 .* l1_deriv;

    dw2 = forward(err1',  l1);         #

    dw1 = forward(err2b', data);       #

    return (dw2 ./ n, dw1 ./ n, err1);

    end;



  function batchn(n, b, i)

    x = b*(i-1) % n;

    return x+1:min(x+b, n);

    end;



  function update(w, data, iters, lr, bat=100)

    n = size(data,1);

    x = floor(iters / 25);

    for iter = 1:iters

      subd = data[batchn(n, bat, iter), :];

      w_d = step(subd, w);

      w = w .+ w_d .* (lr / bat);

      if (iter%x) == 1

        print(iter, "/", iters, "  ", floor(iter*100 / iters), "% complete \n");
 
        end;

      end;

    return w;

    end;



  function ae_update(w, w2, data, iters, lr, bat=100)

    n = size(data,1);

    x = floor(iters / 25);

    for iter = 1:iters

      subd = data[batchn(n, bat, iter), :];

      w_d1, w_d2, errx = ae_Step(subd, w, w2);

      w = w .+ w_d1' .* lr;

      w2 = w .+ w_d2 .* lr;

      if (iter%x) == 1

        print(iter, "/", iters, "  ", floor(iter*100 / iters), "% complete ", mean(abs.(errx[:])), " mae \n");
 
        end;

      end;

    return w, w2;

    end;




  function putweights(f, x)

    CSV.write(f, DataFrame(x, :auto));

    end;


  function getweights(f)

    a = CSV.read(f, DataFrame);

    return Matrix(a[!, :]);

    end;


end;

