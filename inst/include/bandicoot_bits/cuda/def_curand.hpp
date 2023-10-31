// Copyright 2023 Ryan Curtin (http://www.ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



extern "C"
  {



  //
  // setup/teardown functions
  //



  extern curandStatus_t coot_wrapper(curandCreateGenerator)(curandGenerator_t* generator, curandRngType_t rng_type);
  extern curandStatus_t coot_wrapper(curandDestroyGenerator)(curandGenerator_t generator);
  extern curandStatus_t coot_wrapper(curandSetPseudoRandomGeneratorSeed)(curandGenerator_t generator, unsigned long long seed);



  //
  // generation functions
  //



  extern curandStatus_t coot_wrapper(curandGenerate)(curandGenerator_t generator,
                                                     unsigned int* outputPtr,
                                                     size_t num);



  extern curandStatus_t coot_wrapper(curandGenerateUniform)(curandGenerator_t generator,
                                                            float* outputPtr,
                                                            size_t num);



  extern curandStatus_t coot_wrapper(curandGenerateUniformDouble)(curandGenerator_t generator,
                                                                  double* outputPtr,
                                                                  size_t num);



  extern curandStatus_t coot_wrapper(curandGenerateNormal)(curandGenerator_t generator,
                                                           float* outputPtr,
                                                           size_t n,
                                                           float mean,
                                                           float stddev);



  extern curandStatus_t coot_wrapper(curandGenerateNormalDouble)(curandGenerator_t generator,
                                                                 double* outputPtr,
                                                                 size_t n,
                                                                 double mean,
                                                                 double stddev);



  }
